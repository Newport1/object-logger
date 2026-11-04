# Object Logger — submission copy (draft, edit numbers after the real-scene run)

**Track:** Indie — "Build for Your World"

## What it is
Walk a shelf, aisle, bay, or pile once with a phone camera. Get back a numbered master image, a clean crop of every
item, and an editable log for all of them. No photographing items one by one.

## Why
Cataloging groups of physical things is manual: shoot each item, rename files, build a spreadsheet, keep photos and
rows matched. I inventory hundreds of collectibles item-by-item for resale, and only ~300 of thousands of archive photos
are cataloged. Small stores, warehouses, estate sales and resellers all have the same problem.

## How it uses SAM 3.1
- One Meta Model API `sam-3.1` image call on a master frame with a user-editable noun phrase ("toy", "box", "product").
- Boxes + pixel masks decoded with `meta-sam-parser`, filtered (min area, IoU dedupe), numbered in reading order.
- Masks drive the numbered overlay and the per-item crops; confidence is kept on every row.

### Why a master frame instead of raw video tracking
The hosted video route caps tracked objects at 16 per frame and gives no canonical view per item. Collapsing the walk
into one spatial master (panorama when the stitch passes a quality gate, sharpest single frame otherwise) dedupes by
construction, lifts the 16-object cap, and gives a human one image to verify at a glance. Video tracking stays a
future assist for dedupe, never the catalog path.

## Modular by design
Three decoupled layers, each replaceable, joined by one file-based log (`log.json` / `log.csv`):
1. **Capture** — photo, or video → sharp frames → panorama → best-frame fallback. Produces a master frame.
2. **Segmentation** — SAM 3.1 today; any mask provider tomorrow. Produces numbered items.
3. **Enrichment** — v1: human fills name / UPC / price / qty / notes with merge / delete / add fix tools.
   v2 plugs into the same `Item` fields: Muse Spark auto-ID from the crop, barcode read, price lookup, export adapters.

## Demo
15 s walk-past video → master → numbers bloom → click #17 → fix one miss live → type two prices → export CSV.
"15 seconds of video → a catalog."

## Results on the demo scene (fill in)
- Items in scene (hand count): __  · System count before fixes: __ (±__%) · After fixes: exact
- API spend for all dev + demo: $__ of $150

## Run it
```sh
python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
echo MODEL_API_KEY=... > .env
uvicorn app:app --port 8000   # open http://localhost:8000 — the bundled sample session opens with zero API calls
```
