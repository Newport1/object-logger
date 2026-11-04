# Object Logger — build contract (frozen 2026-11-04 10:55 PT)

Deadline: today 6:00 PM PT. Feature freeze 3:00 PM PT. Local web app, no deployment, no build step.
Stack: Python 3.12 venv at `.venv` (already installed: fastapi, uvicorn, python-multipart, pillow, numpy,
opencv-python-headless, meta-sam-parser 0.1.0, requests, openai). Frontend: ONE file `static/index.html`,
vanilla JS + CSS, no framework, no bundler, no CDN dependency.

Run: `. .venv/bin/activate && uvicorn app:app --reload --port 8000` → http://localhost:8000

## Files (fewest possible)

```
app.py            FastAPI: endpoints below + serves static/ and sessions/
pipeline.py       master frame, SAM call, parse, dedupe, sort, crops, overlay, exports
static/index.html the whole UI
tests/test_pipeline.py   ONE test on fixtures/squares.png (makes exactly one API call)
sessions/<id>/    master.png numbered.png crops/NNN.png masks/NNN.png log.json log.csv  (+ source file)
sessions/sample/  pre-processed demo session COMMITTED to git (so judges open the viewer with 0 API calls)
.env              MODEL_API_KEY=...   (gitignored; never log it, never put it in a response body)
```

## SAM 3.1 call — VERIFIED WORKING on this box today. Use this exactly. Do not improvise the API shape.

```python
import os, io, json, base64, requests
from PIL import Image
from meta_sam_parser import image_segmentation_format, CompletedOutcome, decode_mask_to_raster

def load_key():
    k = os.environ.get("MODEL_API_KEY")
    if not k and os.path.exists(".env"):
        for line in open(".env"):
            if line.startswith("MODEL_API_KEY="): k = line.split("=",1)[1].strip()
    if not k: raise RuntimeError("MODEL_API_KEY missing (.env)")
    return k

def segment(master: Image.Image, prompt: str):
    """Returns list of dicts {bbox:[x1,y1,x2,y2] half-open ints, confidence: float|None, mask: np.uint8 HxW full-frame 0/1}."""
    buf = io.BytesIO(); master.save(buf, "PNG")
    url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    body = {"model": "sam-3.1", "stream": False,
            "metadata": {"include_confidence": "true", "mask_encoding": "one_bit"},
            "input": [{"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": url}]}]}
    r = requests.post("https://api.meta.ai/v1/responses",
                      headers={"Authorization": f"Bearer {load_key()}"}, json=body, timeout=180)
    r.raise_for_status()
    j = r.json()
    text = "".join(c.get("text", "") for o in j.get("output", []) for c in o.get("content", [])
                   if c.get("type") == "output_text")
    p = image_segmentation_format().create_parser()
    list(p.push(text))
    result = p.finish(CompletedOutcome()).result   # ImageSegmentationResult: .records .diagnostics .raw_output
    boxes = {r.object_id: r for r in result.records if r.kind == "box"}
    masks = {r.object_id: r for r in result.records if r.kind == "mask"}
    out = []
    for oid, b in boxes.items():
        m = masks.get(oid)
        item = {"bbox": [b.left, b.top, b.right, b.bottom], "confidence": b.confidence, "mask": None}
        if m is not None:
            ras = decode_mask_to_raster(m.mask)            # bytes of 0/1, row-major, BOX-LOCAL size
            h, w = m.mask.height, m.mask.width             # == (bottom-top, right-left)
            local = np.frombuffer(ras, dtype=np.uint8).reshape(h, w)
            full = np.zeros((master.height, master.width), np.uint8)
            full[m.bounds.top:m.bounds.top+h, m.bounds.left:m.bounds.left+w] = local
            item["mask"] = full
        out.append(item)
    return out
```

Facts a builder will NOT know (post-training-cutoff API) — treat as law:
- Response `output_text` is special-token text like
  `<0f>0<|box;x1=100;y1=20;x2=180;y2=100;w=200;h=120;c=0.92|><|mask;x=0;y=0;c=0.92;data=81,81,!...|>,1<|box...`.
  Only meta-sam-parser decodes it. Never write your own parser.
- Records alternate box/mask; pair by `object_id`, not position. `right`/`bottom` are half-open.
- Mask raster is BOX-LOCAL (size = box h × box w), not full frame. Composite at (left, top).
- Coordinates are in pixels of the image you SENT. Downscale master so long edge ≤ 2048 BEFORE sending and
  save that resized image as the canonical `master.png`. All bboxes/crops/overlay use that one size.
- One noun phrase per request ("toy", "box", "product"). Lists like "cup, bottle" return nothing.
- Empty `output_text` = zero matches = valid, not an error. Surface "0 items found, try another prompt".
- Cost $2.50/1000 images. Don't loop it. Fixture test = 1 call.
- Never print or return MODEL_API_KEY.

## Pipeline (pipeline.py)

1. `build_master(path, source_type) -> (PIL.Image master, note:str)`
   - image: open, `ImageOps.exif_transpose`, downscale long edge ≤ 2048, note = "photo".
   - video: `ffmpeg -i in -vf fps=2 -q:v 2 frames/%04d.jpg`; sharpness = `cv2.Laplacian(gray, cv2.CV_64F).var()`.
     Try panorama: take up to 12 frames evenly spaced among the sharper half, `cv2.Stitcher_create(cv2.Stitcher_PANORAMA).stitch(frames)`,
     accept only if status == cv2.Stitcher_OK and 1.2*frame_w <= pano_w <= 6*frame_w and pano_h >= 0.5*frame_h;
     then crop black borders (threshold mask → largest bounding rect), note = f"panorama from {n} frames".
     Else best single frame, note = f"best frame ({sharpness:.0f}) of {N}; panorama fallback: <reason>".
     Whole video step wrapped in try/except → fallback. Fallback is a feature, label it, never 500.
     Do image path FIRST and commit before touching video.
2. `segment(master, prompt)` as above.
3. Post-process: drop area < 0.1% of frame; NMS: IoU > 0.85 keep higher confidence (None counts as 0);
   reading order: band rows by y-center using bin height = median box height, sort by (band, x1); number from 1.
4. Per item: `crops/NNN.png` = bbox + 5% padding (clamped). `masks/NNN.png` = full-frame 0/255 mask (for the
   detail view / numbered overlay). NNN zero-padded 3 digits.
5. `render_numbered(session) -> numbered.png`: master + semi-transparent colored mask fill (if mask) + bbox + number
   badge (white text on dark pill, top-left of bbox). Skip status=deleted. Re-render after every mutation.
6. `write_log(session)`: `log.json` (the Session object below) and `log.csv`
   (columns: number,name,barcode_upc,price,quantity,notes,status,confidence,x1,y1,x2,y2,crop_file). Deleted rows excluded from CSV.
7. `export_zip(session)` → numbered.png + master.png + crops/ + log.json + log.csv.

## Data model (log.json) — frozen; this is the modular interface

```json
Session {
  "id": "20261104-105501-ab12", "created_at": "ISO8601", "source_type": "image|video",
  "source_file": "source.jpg", "concept_prompt": "toy", "master_image": "master.png",
  "numbered_image": "numbered.png", "master_note": "best frame (812) of 31; panorama fallback: status 1",
  "width": 2048, "height": 1152, "item_count": 37,          // non-deleted items
  "items": [ Item ]
}
Item {
  "number": 17, "bbox": [x1, y1, x2, y2], "confidence": 0.91, "mask_ref": "masks/017.png", "crop_file": "crops/017.png",
  "name": "", "barcode_upc": "", "price": "", "notes": "", "quantity": 1,
  "status": "auto|merged|manual|deleted",
  "id_label": "", "id_confidence": null                    // v2 enrichment layer, empty in MVP
}
```
Numbers are stable for a session's life (never renumbered; deleted items keep their number, hidden everywhere).

## HTTP API (app.py) — frontend and backend builders both bind to this exactly

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/` | | static/index.html |
| GET | `/api/sessions` | | `[{id, created_at, concept_prompt, item_count, source_type}]` newest first |
| POST | `/api/sessions` | multipart `file` (jpg/png/mp4/mov), `prompt` (default "product") | Session (201). Errors: 400 bad file/prompt, 502 SAM failure with `{"detail": msg}` |
| GET | `/api/sessions/{id}` | | Session |
| POST | `/api/sessions/{id}/resegment` | JSON `{"prompt": "toy"}` | Session — re-runs SAM on existing master; replaces ALL items (manual edits lost; UI must confirm) |
| PATCH | `/api/sessions/{id}/items/{number}` | JSON any of `name, barcode_upc, price, notes, quantity` | Session |
| DELETE | `/api/sessions/{id}/items/{number}` | | Session (status→deleted) |
| POST | `/api/sessions/{id}/items/merge` | JSON `{"numbers": [3, 4]}` (≥2) | Session — keeps lowest number, bbox = union, mask = OR, crop regenerated, status "merged", others → deleted, notes gets "merged from 3,4" |
| POST | `/api/sessions/{id}/items` | JSON `{"bbox": [x1,y1,x2,y2]}` in master pixels | Session — new item number = max+1, status "manual", no mask, crop from bbox |
| GET | `/api/sessions/{id}/export.json` | | log.json download |
| GET | `/api/sessions/{id}/export.csv` | | log.csv download |
| GET | `/api/sessions/{id}/export.zip` | | zip download |
| GET | `/sessions/{id}/master.png`, `/sessions/{id}/numbered.png`, `/sessions/{id}/crops/NNN.png` | | static files (mount `sessions/` at `/sessions`) |

Every mutating call re-renders numbered.png and rewrites log.json/log.csv before returning. Session id must match
`^[A-Za-z0-9_-]+$` (path-traversal guard). Uploads > 200 MB → 413.

## Frontend (static/index.html) — hero screen is the numbered master

- Top bar: file input (image/video) + prompt text input (default "product") + "Scan" button + session dropdown
  (loads `/api/sessions`; selecting loads that session — the committed `sample` session must open with zero API calls).
  Show a spinner + "Segmenting…" while POST runs (can take 10–60 s for video). Show `master_note` under the image.
- Hero: `<img src=master.png>` inside `position:relative` div; overlay boxes + number badges drawn as absolutely
  positioned divs from `items` (scale = displayed width / session.width). Don't use numbered.png in the UI (that's for export).
  Click a box → highlight it + highlight and scroll to its row. Click a row → highlight its box.
- Count headline: "37 items" (item_count), updates after every mutation.
- Log grid: one row per non-deleted item: #, crop thumbnail (`crops/NNN.png`, click = large view), status, inputs for
  name / barcode_upc / price / quantity / notes. Save on `change` via PATCH (no save button). Delete button per row.
- Fix tools: checkbox per row + "Merge selected" button (POST merge). "Add item" toggle: drag a rectangle on the
  master image → POST items with bbox in master pixels (`Math.floor((clientX-rect.left)*session.width/rect.width)`).
  "Re-run prompt" button → confirm() → POST resegment.
- Export: three links (`export.csv`, `export.json`, `export.zip`).
- Plain, readable, dark-on-light, works at 1280 px wide. No design heroics. Must work in Chromium and Firefox.

## Test (tests/test_pipeline.py) — the one check that must pass

Run `. .venv/bin/activate && python -m pytest -q tests` (or `python tests/test_pipeline.py`). Uses `fixtures/squares.png`
(400×240: red square at [20,20,120,120], blue [200,40,360,200], green [140,150,190,220]), prompt "square", ONE API call.
Assert: 2–3 items; numbering is reading order (red is #1); each crop file exists and its size ≈ bbox+padding;
log.json loads and has the Session keys; log.csv has item_count data rows; numbered.png exists and equals master size.
Then exercise merge(1,2) and delete and add via the pipeline functions (no API) and assert item_count and statuses.

## Fences
- Never commit `.env`, any session except `sessions/sample/`, or the venv.
- Don't add dependencies beyond requirements.txt without saying so.
- Commit after each demoable step with a conventional message.
