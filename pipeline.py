import os, io, re, csv, json, base64, shutil, zipfile, subprocess, tempfile, datetime, secrets, sys, multiprocessing
from pathlib import Path

import numpy as np
import requests
import cv2
from PIL import Image, ImageOps, ImageDraw, ImageFont
from meta_sam_parser import image_segmentation_format, CompletedOutcome, decode_mask_to_raster

ROOT = Path(__file__).parent
SESSIONS = ROOT / "sessions"
MAX_EDGE = 2048
ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
FIELDS = ("name", "barcode_upc", "price", "notes", "quantity")


# ---------------------------------------------------------------- SAM (contract snippet)
def load_key():
    k = os.environ.get("MODEL_API_KEY")
    env = ROOT / ".env"
    if not k and env.exists():
        for line in open(env):
            if line.startswith("MODEL_API_KEY="):
                k = line.split("=", 1)[1].strip()
    if not k:
        raise RuntimeError("MODEL_API_KEY missing (.env)")
    return k


def segment(master: Image.Image, prompt: str):
    """Returns list of dicts {bbox, confidence, mask(full-frame 0/1 uint8 or None)}."""
    buf = io.BytesIO(); master.save(buf, "PNG")
    url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    body = {"model": "sam-3.1", "stream": False,
            "metadata": {"include_confidence": "true", "mask_encoding": "one_bit"},
            "input": [{"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": url}]}]}
    key = load_key()
    r = requests.post("https://api.meta.ai/v1/responses",
                      headers={"Authorization": f"Bearer {key}"}, json=body, timeout=180)
    r.raise_for_status()
    j = r.json()
    text = "".join(c.get("text", "") for o in j.get("output", []) for c in o.get("content", [])
                   if c.get("type") == "output_text")
    p = image_segmentation_format().create_parser()
    list(p.push(text))
    result = p.finish(CompletedOutcome()).result
    boxes = {r.object_id: r for r in result.records if r.kind == "box"}
    masks = {r.object_id: r for r in result.records if r.kind == "mask"}
    out = []
    for oid, b in boxes.items():
        m = masks.get(oid)
        item = {"bbox": [b.left, b.top, b.right, b.bottom], "confidence": b.confidence, "mask": None}
        if m is not None:
            ras = decode_mask_to_raster(m.mask)
            h, w = m.mask.height, m.mask.width
            local = np.frombuffer(ras, dtype=np.uint8).reshape(h, w)
            full = np.zeros((master.height, master.width), np.uint8)
            full[m.bounds.top:m.bounds.top + h, m.bounds.left:m.bounds.left + w] = local
            item["mask"] = full
        out.append(item)
    return out


# ---------------------------------------------------------------- master frame
def _downscale(img: Image.Image) -> Image.Image:
    s = MAX_EDGE / max(img.size)
    if s < 1:
        img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.LANCZOS)
    return img


def _sharp(bgr):
    return cv2.Laplacian(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()


def _crop_black(pano):
    gray = cv2.cvtColor(pano, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(gray > 8)
    if len(xs) == 0:
        return pano
    mask = (gray > 8).astype(np.uint8)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    return pano[y:y + h, x:x + w]


STITCH_TIMEOUT = 90  # seconds; repetitive texture (a wall of identical boxes) can make feature matching run for hours


def _stitch_worker(paths, out):
    frames = [cv2.imread(p) for p in paths]
    try:
        st = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    except AttributeError:
        st = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)
    status, pano = st.stitch(frames)
    if status == cv2.Stitcher_OK:
        cv2.imwrite(out, pano)
    sys.exit(int(status))


def _stitch_with_timeout(paths, out):
    # ponytail: cv2 can't be interrupted in-thread, so stitch in a child process we can kill
    ctx = multiprocessing.get_context("spawn")
    p = ctx.Process(target=_stitch_worker, args=(paths, out), daemon=True)
    p.start()
    p.join(STITCH_TIMEOUT)
    if p.is_alive():
        p.kill()
        p.join()
        raise RuntimeError(f"stitch timed out after {STITCH_TIMEOUT} s")
    if p.exitcode != 0 or not os.path.exists(out):
        raise RuntimeError(f"stitch status {p.exitcode}")
    return cv2.imread(out)


def build_master(path, source_type, force_fallback=False):
    path = str(path)
    if source_type == "image":
        img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        return _downscale(img), "photo"
    # video
    with tempfile.TemporaryDirectory() as td:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", path, "-vf", "fps=2", "-q:v", "2",
                        os.path.join(td, "%04d.jpg")], check=True, timeout=300)
        files = sorted(Path(td).glob("*.jpg"))
        if not files:
            raise ValueError("no frames could be extracted from video")
        frames = [cv2.imread(str(f)) for f in files]
        N = len(frames)
        sharp = [_sharp(f) for f in frames]
        best_i = int(np.argmax(sharp))
        reason = "unknown"
        try:
            if force_fallback:
                raise RuntimeError("forced fallback")
            order = sorted(range(N), key=lambda i: -sharp[i])[:max(2, (N + 1) // 2)]
            order = sorted(order)  # temporal order among sharper half
            k = min(12, len(order))
            pick = [order[int(round(i))] for i in np.linspace(0, len(order) - 1, k)]
            pick = sorted(set(pick))
            sel = [frames[i] for i in pick]
            if len(sel) < 2:
                raise RuntimeError("too few frames")
            pano = _stitch_with_timeout([str(files[i]) for i in pick], os.path.join(td, "pano.png"))
            pano = _crop_black(pano)
            fh, fw = sel[0].shape[:2]
            ph, pw = pano.shape[:2]
            if not (1.2 * fw <= pw <= 6 * fw and ph >= 0.5 * fh):
                raise RuntimeError(f"gate failed: pano {pw}x{ph} vs frame {fw}x{fh}")
            img = Image.fromarray(cv2.cvtColor(pano, cv2.COLOR_BGR2RGB))
            return _downscale(img), f"panorama from {len(sel)} frames"
        except Exception as e:  # fallback is a feature
            reason = str(e) or type(e).__name__
        img = Image.fromarray(cv2.cvtColor(frames[best_i], cv2.COLOR_BGR2RGB))
        return _downscale(img), f"best frame ({sharp[best_i]:.0f}) of {N}; panorama fallback: {reason}"


# ---------------------------------------------------------------- post-process
def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def postprocess(dets, W, H):
    dets = [d for d in dets if (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]) >= 0.001 * W * H]
    dets.sort(key=lambda d: -(d["confidence"] or 0))
    kept = []
    for d in dets:
        if all(_iou(d["bbox"], k["bbox"]) <= 0.85 for k in kept):
            kept.append(d)
    if not kept:
        return []
    bin_h = max(1.0, float(np.median([d["bbox"][3] - d["bbox"][1] for d in kept])))
    # ponytail: greedy row clustering on center-y; fixed bins split a row whenever it straddled a bin edge
    cy = lambda d: (d["bbox"][1] + d["bbox"][3]) / 2
    rows = []
    for d in sorted(kept, key=cy):
        if rows and cy(d) - cy(rows[-1][0]) <= 0.5 * bin_h:
            rows[-1].append(d)
        else:
            rows.append([d])
    return [d for row in rows for d in sorted(row, key=lambda d: d["bbox"][0])]


# ---------------------------------------------------------------- session files
def sdir(sid):
    if not ID_RE.match(sid or ""):
        raise ValueError("bad session id")
    return SESSIONS / sid


def load_session(sid):
    p = sdir(sid) / "log.json"
    if not p.exists():
        raise FileNotFoundError(sid)
    return json.loads(p.read_text())


def _write_crop(session, it):
    d = sdir(session["id"])
    master = Image.open(d / session["master_image"]).convert("RGB")
    x1, y1, x2, y2 = it["bbox"]
    px, py = round((x2 - x1) * 0.05), round((y2 - y1) * 0.05)
    box = (max(0, x1 - px), max(0, y1 - py), min(master.width, x2 + px), min(master.height, y2 + py))
    (d / "crops").mkdir(exist_ok=True)
    master.crop(box).save(d / it["crop_file"])


def _write_mask(session, it, arr):
    d = sdir(session["id"])
    (d / "masks").mkdir(exist_ok=True)
    Image.fromarray((arr > 0).astype(np.uint8) * 255).save(d / it["mask_ref"])


def _fresh_item(n, bbox, conf, status, has_mask):
    return {"number": n, "bbox": [int(v) for v in bbox], "confidence": conf,
            "mask_ref": f"masks/{n:03d}.png" if has_mask else None, "crop_file": f"crops/{n:03d}.png",
            "name": "", "barcode_upc": "", "price": "", "notes": "", "quantity": 1,
            "status": status, "id_label": "", "id_confidence": None}


def live(session):
    return [i for i in session["items"] if i["status"] != "deleted"]


def _color(n):
    r = np.random.RandomState(n * 7919 % 2**31)
    return tuple(int(v) for v in r.randint(40, 230, 3))


def render_numbered(session):
    d = sdir(session["id"])
    base = Image.open(d / session["master_image"]).convert("RGBA")
    over = Image.new("RGBA", base.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(over)
    font = ImageFont.load_default(size=max(14, base.width // 70))
    for it in live(session):
        col = _color(it["number"])
        if it.get("mask_ref") and (d / it["mask_ref"]).exists():
            m = np.array(Image.open(d / it["mask_ref"]).convert("L")) > 0
            layer = np.zeros((base.height, base.width, 4), np.uint8)
            layer[m] = (*col, 110)
            over = Image.alpha_composite(over, Image.fromarray(layer, "RGBA"))
            od = ImageDraw.Draw(over)
        x1, y1, x2, y2 = it["bbox"]
        od.rectangle([x1, y1, x2 - 1, y2 - 1], outline=(*col, 255), width=3)
        t = str(it["number"])
        tb = od.textbbox((0, 0), t, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        od.rounded_rectangle([x1, y1, x1 + tw + 12, y1 + th + 10], radius=6, fill=(20, 20, 20, 230))
        od.text((x1 + 6 - tb[0], y1 + 5 - tb[1]), t, fill=(255, 255, 255, 255), font=font)
    Image.alpha_composite(base, over).convert("RGB").save(d / session["numbered_image"])


def write_log(session):
    d = sdir(session["id"])
    session["item_count"] = len(live(session))
    (d / "log.json").write_text(json.dumps(session, indent=2))
    with open(d / "log.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["number", "name", "barcode_upc", "price", "quantity", "notes", "status", "confidence",
                    "x1", "y1", "x2", "y2", "crop_file"])
        for it in live(session):
            w.writerow([it["number"], it["name"], it["barcode_upc"], it["price"], it["quantity"], it["notes"],
                        it["status"], it["confidence"], *it["bbox"], it["crop_file"]])


def save(session, render=False):
    # ponytail: numbered.png costs ~1.7 s on a 2048px master, so item edits skip it; export renders fresh
    if render:
        render_numbered(session)
    write_log(session)
    return session


def _populate(session, master, prompt):
    """Run SAM on master, replace all items, write crops/masks."""
    d = sdir(session["id"])
    dets = postprocess(segment(master, prompt), master.width, master.height)
    for sub in ("crops", "masks"):
        shutil.rmtree(d / sub, ignore_errors=True)
        (d / sub).mkdir()
    items = []
    for n, det in enumerate(dets, 1):
        it = _fresh_item(n, det["bbox"], det["confidence"], "auto", det["mask"] is not None)
        _write_crop(session, it)
        if det["mask"] is not None:
            _write_mask(session, it, det["mask"])
        items.append(it)
    session["items"] = items
    session["concept_prompt"] = prompt
    return save(session, render=True)


def create_session(src_path, source_type, prompt, sid=None, force_fallback=False):
    sid = sid or datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2)
    d = sdir(sid)
    d.mkdir(parents=True, exist_ok=True)
    ext = Path(src_path).suffix.lower() or (".png" if source_type == "image" else ".mp4")
    shutil.copy(src_path, d / f"source{ext}")
    try:
        master, note = build_master(src_path, source_type, force_fallback=force_fallback)
        master.save(d / "master.png")
        session = {"id": sid, "created_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                   "source_type": source_type, "source_file": f"source{ext}", "concept_prompt": prompt,
                   "master_image": "master.png", "numbered_image": "numbered.png", "master_note": note,
                   "width": master.width, "height": master.height, "item_count": 0, "items": []}
        return _populate(session, master, prompt)
    except Exception:
        shutil.rmtree(d, ignore_errors=True)  # no half-made sessions on failure
        raise


def resegment(sid, prompt):
    session = load_session(sid)
    master = Image.open(sdir(sid) / session["master_image"]).convert("RGB")
    return _populate(session, master, prompt)


# ---------------------------------------------------------------- mutations (no API)
def _find(session, number):
    for it in session["items"]:
        if it["number"] == number:
            return it
    raise KeyError(number)


def patch_item(session, number, fields):
    it = _find(session, number)
    for k in FIELDS:
        if k in fields:
            v = fields[k]
            if k == "quantity":
                v = int(v)
            else:
                v = "" if v is None else str(v)
            it[k] = v
    return save(session)


def delete_item(session, number):
    _find(session, number)["status"] = "deleted"
    return save(session)


def merge_items(session, numbers):
    nums = sorted(set(int(n) for n in numbers))
    if len(nums) < 2:
        raise ValueError("need at least 2 distinct items to merge")
    items = [_find(session, n) for n in nums]
    if any(i["status"] == "deleted" for i in items):
        raise ValueError("cannot merge deleted items")
    d = sdir(session["id"])
    keep = items[0]
    keep["bbox"] = [min(i["bbox"][0] for i in items), min(i["bbox"][1] for i in items),
                    max(i["bbox"][2] for i in items), max(i["bbox"][3] for i in items)]
    acc = None
    for i in items:
        if i.get("mask_ref") and (d / i["mask_ref"]).exists():
            m = np.array(Image.open(d / i["mask_ref"]).convert("L")) > 0
            acc = m if acc is None else (acc | m)
    keep["mask_ref"] = f"masks/{keep['number']:03d}.png" if acc is not None else None
    if acc is not None:
        _write_mask(session, keep, acc)
    keep["status"] = "merged"
    keep["notes"] = (keep["notes"] + "; " if keep["notes"] else "") + "merged from " + ",".join(map(str, nums))
    confs = [i["confidence"] for i in items if i["confidence"] is not None]
    keep["confidence"] = max(confs) if confs else None
    for i in items[1:]:
        i["status"] = "deleted"
    _write_crop(session, keep)
    return save(session)


def add_manual(session, bbox):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, x2 = sorted((max(0, x1), min(session["width"], x2)))
    y1, y2 = sorted((max(0, y1), min(session["height"], y2)))
    if x2 - x1 < 2 or y2 - y1 < 2:
        raise ValueError("bbox too small / outside image")
    n = max([i["number"] for i in session["items"]], default=0) + 1
    it = _fresh_item(n, [x1, y1, x2, y2], None, "manual", False)
    session["items"].append(it)
    _write_crop(session, it)
    return save(session)


def export_zip(session):
    d = sdir(session["id"])
    out = d / "export.zip"
    render_numbered(session)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in ("numbered.png", "master.png", "log.json", "log.csv"):
            z.write(d / f, f)
        for it in live(session):
            z.write(d / it["crop_file"], it["crop_file"])
    return out
