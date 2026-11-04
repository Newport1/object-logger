import json, os, shutil, subprocess, tempfile
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import pipeline as P

app = FastAPI()
MAX_BYTES = 200 * 1024 * 1024
IMG = {".jpg", ".jpeg", ".png"}
VID = {".mp4", ".mov"}
P.SESSIONS.mkdir(exist_ok=True)


def err(code, msg):
    return HTTPException(code, detail=msg)


def clean_prompt(p):
    p = (p or "").strip()
    if not p or len(p) > 100:
        raise err(400, "prompt must be 1-100 characters")
    return p


def get(sid):
    try:
        return P.load_session(sid)
    except ValueError:
        raise err(400, "bad session id")
    except FileNotFoundError:
        raise err(404, "session not found")


def sam_failure(e):
    msg = str(e)
    try:
        msg = msg.replace(P.load_key(), "***")
    except Exception:
        pass
    return err(502, f"segmentation failed: {msg[:300]}")


@app.get("/api/sessions")
def list_sessions():
    rows = []
    for d in P.SESSIONS.iterdir():
        f = d / "log.json"
        if d.is_dir() and f.exists():
            try:
                s = json.loads(f.read_text())
                rows.append({k: s[k] for k in ("id", "created_at", "concept_prompt", "item_count", "source_type")})
            except Exception:
                pass
    rows.sort(key=lambda r: r["created_at"], reverse=True)
    return rows


@app.post("/api/sessions", status_code=201)
def create(file: UploadFile = File(...), prompt: str = Form("product")):
    prompt = clean_prompt(prompt)
    ext = Path(file.filename or "").suffix.lower()
    if ext in IMG:
        st = "image"
    elif ext in VID:
        st = "video"
    else:
        raise err(400, "file must be jpg/png/mp4/mov")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    try:
        size = 0
        while chunk := file.file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_BYTES:
                raise err(413, "file too large (max 200 MB)")
            tmp.write(chunk)
        tmp.close()
        try:
            return P.create_session(tmp.name, st, prompt)
        except HTTPException:
            raise
        except Exception as e:
            if st == "image" and "cannot identify image" in str(e):
                raise err(400, "could not read image")
            if isinstance(e, (OSError, ValueError, subprocess.SubprocessError)) and not hasattr(e, "response"):
                raise err(400, f"bad file: {str(e)[:200]}")
            raise sam_failure(e)
    finally:
        tmp.close()
        os.unlink(tmp.name)


@app.get("/api/sessions/{sid}")
def get_session(sid: str):
    return get(sid)


class Prompt(BaseModel):
    prompt: str


@app.post("/api/sessions/{sid}/resegment")
def resegment(sid: str, body: Prompt):
    get(sid)
    try:
        return P.resegment(sid, clean_prompt(body.prompt))
    except HTTPException:
        raise
    except Exception as e:
        raise sam_failure(e)


class Patch(BaseModel):
    name: str | None = None
    barcode_upc: str | None = None
    price: str | None = None
    notes: str | None = None
    quantity: int | None = None


def mutate(sid, fn):
    s = get(sid)
    try:
        return fn(s)
    except KeyError:
        raise err(404, "item not found")
    except (ValueError, TypeError) as e:
        raise err(400, str(e))


# NOTE: static path /items/merge and POST /items must be declared before /items/{number} routes (different verbs, fine)
class Merge(BaseModel):
    numbers: List[int]


@app.post("/api/sessions/{sid}/items/merge")
def merge(sid: str, body: Merge):
    return mutate(sid, lambda s: P.merge_items(s, body.numbers))


class Add(BaseModel):
    bbox: List[int]


@app.post("/api/sessions/{sid}/items")
def add(sid: str, body: Add):
    if len(body.bbox) != 4:
        raise err(400, "bbox must be [x1,y1,x2,y2]")
    return mutate(sid, lambda s: P.add_manual(s, body.bbox))


@app.patch("/api/sessions/{sid}/items/{number}")
def patch(sid: str, number: int, body: Patch):
    fields = body.model_dump(exclude_unset=True)
    return mutate(sid, lambda s: P.patch_item(s, number, fields))


@app.delete("/api/sessions/{sid}/items/{number}")
def delete(sid: str, number: int):
    return mutate(sid, lambda s: P.delete_item(s, number))


@app.get("/api/sessions/{sid}/export.json")
def export_json(sid: str):
    get(sid)
    return FileResponse(P.sdir(sid) / "log.json", media_type="application/json", filename=f"{sid}.json")


@app.get("/api/sessions/{sid}/export.csv")
def export_csv(sid: str):
    get(sid)
    return FileResponse(P.sdir(sid) / "log.csv", media_type="text/csv", filename=f"{sid}.csv")


@app.get("/api/sessions/{sid}/export.zip")
def export_zip(sid: str):
    s = get(sid)
    return FileResponse(P.export_zip(s), media_type="application/zip", filename=f"{sid}.zip")


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


app.mount("/sessions", StaticFiles(directory=str(P.SESSIONS)), name="sessions")
