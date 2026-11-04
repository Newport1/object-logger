# Object Logger

Walk a shelf, aisle, bay, or pile once with a camera. Get back a numbered master image, a crop of every item,
and an editable log. Built on SAM 3.1 via the Meta Model API for the Meta hackathon (Indie track, Nov 2026).

## Run

```sh
python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
echo "MODEL_API_KEY=<your key from https://dev.meta.ai/platform>" > .env
uvicorn app:app --port 8000
```

Open http://localhost:8000. The bundled `sessions/sample/` (37 items) opens with zero API calls.
ffmpeg must be on PATH for video input.

## Use

1. Choose a photo (jpg/png) or a short lateral walk-past video (mp4/mov), type one noun phrase
   ("object", "toy", "can", "product"), click **Scan**. Images take ~10–20 s, video up to a minute.
2. The numbered master is the hero: click a number to jump to its row, click a row to light its box.
3. Fix the count: **Delete** a bad segment, tick two rows and **Merge selected**, or **Add item** and drag a box
   on the master for anything SAM missed. **Re-run prompt** re-segments with a new phrase.
4. Fill name / UPC / price / qty / notes inline. Export **CSV**, **JSON**, or a **ZIP** with crops + master.

The prompt matters more than anything else: on a supermarket aisle, "product" returned 23 cans and "can" returned 99.
Re-running is one click and about a quarter of a cent.

## How it works

```
photo or video ──► master frame ──► SAM 3.1 (image mode) ──► numbered items ──► editable log ──► CSV / JSON / ZIP
                   video: ffmpeg 2 fps → Laplacian sharpness → OpenCV panorama (quality-gated) → else best frame
```

- `pipeline.py` — master frame, SAM call, mask decode (`meta-sam-parser`), area filter + IoU dedupe,
  reading-order numbering, crops, overlay, merge / delete / add, exports.
- `app.py` — FastAPI, one session = one folder (`master.png`, `numbered.png`, `crops/`, `masks/`, `log.json`, `log.csv`).
- `static/index.html` — the whole UI, no build step.
- `tests/test_pipeline.py` — one end-to-end test on `fixtures/squares.png` (one API call).

Why a master frame instead of video tracking: the hosted video route caps at 16 tracked objects per frame and gives no
canonical view per item. One spatial master dedupes by construction, removes the cap, and is verifiable at a glance.

## Modular

Three replaceable layers joined by the `Item` record in `log.json`: capture (photo / video→pano / future tracks),
segmentation (SAM 3.1 today, any mask provider tomorrow), enrichment (human today; `id_label` / `id_confidence`
fields are reserved for Muse Spark auto-ID, barcode read, price lookup, marketplace export adapters).

## Credits

Sample scene: "A colorful retro display…" by [PattayaPatrol](https://www.flickr.com/people/PattayaPatrol/),
via Wikimedia Commons, CC BY-SA 4.0. Segmentation: Meta SAM 3.1. Mask decoding: `meta-sam-parser`.
