# Object Logger

Walk a shelf, aisle, bay, or pile once with a camera. Get back a numbered master image, a crop of every item,
and an editable log. Built on SAM 3.1 via the Meta Model API for the Meta hackathon (Indie track, Nov 2026).

## Run

```sh
python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
echo "MODEL_API_KEY=<your key from https://dev.meta.ai/platform>" > .env
uvicorn app:app --port 8000
```

Open http://localhost:8000. Two bundled sessions open with zero API calls: `sessions/sample/` (a 44-figure
Bearbrick tray from one photo, prompt "toy", 44/44 found) and `sessions/sample-video/` (a 7 s phone walk-past
of a toy shelf, stitched into a panorama, 20 detections for 19 toys).
ffmpeg must be on PATH for video input.

## Use

1. Choose a photo (jpg/png) or a short lateral walk-past video (mp4/mov), type one noun phrase
   ("object", "toy", "can", "product"), click **Scan**. Images take ~10–20 s, video up to a minute.
2. The numbered master is the hero: click a number to jump to its row, click a row to light its box.
3. Fix the count: **Delete** a bad segment, tick two rows and **Merge selected**, or **Add item** and drag a box
   on the master for anything SAM missed. **Re-run prompt** re-segments with a new phrase.
4. Fill name / UPC / price / qty / notes inline. Export **CSV**, **JSON**, or a **ZIP** with crops + master.

The prompt matters more than anything else. Use one concrete noun for the thing on the shelf; generic words often
return nothing. Re-running is one click and about a quarter of a cent. Measured on 12 scenes:

| Scene | Works | Returns 0 or far fewer |
|---|---|---|
| Collectible figures, carved animals, clay dolls | `toy`, `figurine` | `figure`, `object` |
| Supermarket cans | `can` (99) | `product` (23) |
| Flat sardine tins | `package` (72), `product` (38) | `can`, `tin` |
| Snack bags, bottles, manga spines, cameras, film boxes | `bag`, `bottle`, `book`, `camera`, `box` | |

## How it works

```
photo or video ──► master frame ──► SAM 3.1 (image mode) ──► numbered items ──► editable log ──► CSV / JSON / ZIP
                   video: ffmpeg 2 fps → Laplacian sharpness → OpenCV panorama (quality-gated, 90 s cap) → else best frame
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

Sample scenes: the author's own collection. Segmentation: Meta SAM 3.1. Mask decoding: `meta-sam-parser`.
