import csv, json, shutil, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from PIL import Image
import pipeline as P

FIX = Path(__file__).resolve().parent.parent / "fixtures" / "squares.png"


def test_pipeline_end_to_end():
    sid = "test-squares"
    shutil.rmtree(P.SESSIONS / sid, ignore_errors=True)
    try:
        s = P.create_session(FIX, "image", "square", sid=sid)  # the ONE API call
        d = P.SESSIONS / sid
        items = P.live(s)
        assert 2 <= len(items) <= 3, items
        assert [i["number"] for i in items] == list(range(1, len(items) + 1))
        # reading order: red square (top-left) is #1
        x1, y1, x2, y2 = items[0]["bbox"]
        assert abs(x1 - 20) <= 8 and abs(y1 - 20) <= 8 and abs(x2 - 120) <= 8 and abs(y2 - 120) <= 8, items[0]["bbox"]
        for it in items:
            c = Image.open(d / it["crop_file"])
            bx1, by1, bx2, by2 = it["bbox"]
            px, py = round((bx2 - bx1) * 0.05), round((by2 - by1) * 0.05)
            assert abs(c.width - (bx2 - bx1 + 2 * px)) <= 2 * max(px, 1) + 2
            assert abs(c.height - (by2 - by1 + 2 * py)) <= 2 * max(py, 1) + 2
        log = json.loads((d / "log.json").read_text())
        for k in ("id", "created_at", "source_type", "source_file", "concept_prompt", "master_image",
                  "numbered_image", "master_note", "width", "height", "item_count", "items"):
            assert k in log
        assert log["item_count"] == len(items)
        rows = list(csv.DictReader(open(d / "log.csv")))
        assert len(rows) == len(items)
        assert Image.open(d / "numbered.png").size == Image.open(d / "master.png").size

        # mutations, no API
        n0 = len(items)
        s = P.merge_items(s, [1, 2])
        assert s["item_count"] == n0 - 1
        assert s["items"][0]["status"] == "merged" and "merged from 1,2" in s["items"][0]["notes"]
        assert s["items"][1]["status"] == "deleted"
        s = P.delete_item(s, 1)
        assert s["item_count"] == n0 - 2
        s = P.add_manual(s, [10, 200, 60, 230])
        new = s["items"][-1]
        assert new["status"] == "manual" and new["number"] == n0 + 1 and new["mask_ref"] is None
        assert s["item_count"] == n0 - 1
        assert (d / new["crop_file"]).exists()
        s = P.patch_item(s, new["number"], {"name": "thing", "quantity": "3"})
        assert s["items"][-1]["quantity"] == 3
        assert P.export_zip(s).exists()
    finally:
        shutil.rmtree(P.SESSIONS / sid, ignore_errors=True)


if __name__ == "__main__":
    test_pipeline_end_to_end()
    print("ok")
