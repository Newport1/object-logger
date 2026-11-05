import asyncio, time, json, sys
from playwright.async_api import async_playwright
S = "/tmp/claude-1000/-home-user1-developer-musehackathontest/8106e933-f73a-4fda-a8c7-43e8b794d107/scratchpad"
REPO = "/home/user1/developer/musehackathontest"
W, H = 1400, 1500
CURSOR = """
(() => { const c = document.createElement('div'); c.id = 'fakecursor';
  c.style.cssText = 'position:fixed;z-index:99999;width:18px;height:18px;border-radius:50%;background:rgba(220,40,40,.85);border:2px solid #fff;box-shadow:0 0 6px rgba(0,0,0,.5);pointer-events:none;left:-50px;top:-50px;transform:translate(-50%,-50%)';
  const add = () => document.body && document.body.appendChild(c);
  document.readyState === 'loading' ? document.addEventListener('DOMContentLoaded', add) : add();
  document.addEventListener('mousemove', e => { c.style.left = e.clientX + 'px'; c.style.top = e.clientY + 'px'; }, true);
  document.addEventListener('mousedown', () => { c.style.transform = 'translate(-50%,-50%) scale(.7)'; }, true);
  document.addEventListener('mouseup', () => { c.style.transform = 'translate(-50%,-50%)'; }, true);
})();
"""
marks = {}
T0 = None
def mark(k): marks[k] = round(time.time() - T0, 2)

async def glide(pg, x, y, steps=25):
    await pg.mouse.move(x, y, steps=steps)

async def click_at(pg, loc, pause=0.6):
    await loc.scroll_into_view_if_needed(); await asyncio.sleep(0.4)
    b = await loc.bounding_box()
    x, y = b["x"] + b["width"] / 2, b["y"] + b["height"] / 2
    await glide(pg, x, y); await asyncio.sleep(0.25); await pg.mouse.click(x, y); await asyncio.sleep(pause)

async def main():
    global T0
    async with async_playwright() as p:
        b = await p.chromium.launch(executable_path="/usr/bin/google-chrome", headless=True)
        ctx = await b.new_context(viewport={"width": W, "height": H}, record_video_dir=f"{S}/vid", record_video_size={"width": W, "height": H}, accept_downloads=True)
        await ctx.add_init_script(CURSOR)
        pg = await ctx.new_page(); T0 = time.time()
        await pg.goto("http://localhost:8000/"); await pg.wait_for_selector("tbody tr")
        await glide(pg, 400, 300); await asyncio.sleep(1.0)
        # 1. pick the walk-past video, type the prompt, scan
        await pg.set_input_files("#file", f"{REPO}/fixtures/walk_past.mov")
        await click_at(pg, pg.locator("#prompt"), 0.2)
        await pg.keyboard.press("Control+A"); await pg.keyboard.type("toy", delay=120); await asyncio.sleep(0.5)
        mark("scan_click")
        async with pg.expect_response(lambda r: r.url.endswith("/api/sessions") and r.request.method == "POST", timeout=180000) as ri:
            await click_at(pg, pg.locator("#scan"), 0.1)
            await asyncio.sleep(2.5); mark("cut_start")
        r = await ri.value; sid = (await r.json())["id"]
        await pg.wait_for_function("!document.body.innerText.includes('Segmenting')"); await pg.wait_for_selector("#boxes .box")
        mark("cut_end"); await pg.evaluate("window.scrollTo(0,0)"); await asyncio.sleep(2.5)
        await glide(pg, 300, 500, 40); await glide(pg, 1100, 900, 60); await asyncio.sleep(1.0)
        # 2. hover a few boxes, click the fragment (#9), then its parent (#3)
        for n in (12, 5, 9):
            await click_at(pg, pg.locator(f"#boxes .box[data-num='{n}']"), 0.9)
        await asyncio.sleep(0.8)
        # 3. merge 3 + 9
        for n in (3, 9):
            await click_at(pg, pg.locator(f"#rows tr[data-num='{n}'] input[type=checkbox]"), 0.4)
        async with pg.expect_response(lambda r: "/items/merge" in r.url) as ri:
            await click_at(pg, pg.locator("#mergebtn"), 0.2)
        await pg.wait_for_function("!document.body.innerText.includes('Saving')"); await pg.evaluate("window.scrollTo(0,0)"); await asyncio.sleep(2.5)
        # 4. type a name and two prices
        for n, name, price in ((1, "Purple kaiju", "45"), (2, "Blue beast", "60")):
            row = pg.locator(f"#rows tr[data-num='{n}']")
            await click_at(pg, row.locator("input[aria-label^='name']"), 0.1); await pg.keyboard.type(name, delay=70)
            await click_at(pg, row.locator("input[aria-label^='price']"), 0.1); await pg.keyboard.type(price, delay=90)
            await pg.keyboard.press("Tab"); await asyncio.sleep(0.8)
        await pg.wait_for_function("!document.body.innerText.includes('Saving')"); await asyncio.sleep(0.6)
        # 5. export CSV
        async with pg.expect_download() as dl:
            await click_at(pg, pg.locator("#x-csv"), 0.3)
        await (await dl.value).save_as(f"{S}/vid/demo_export.csv"); await asyncio.sleep(1.2)
        # 6. switch to the photo sample (44/44)
        await click_at(pg, pg.locator("#sessions"), 0.2)
        await pg.select_option("#sessions", "sample"); await pg.keyboard.press("Escape")
        await pg.wait_for_function("document.querySelectorAll('#boxes .box').length >= 40"); await pg.evaluate("window.scrollTo(0,0)"); await asyncio.sleep(2.5)
        await glide(pg, 700, 700, 50); await pg.mouse.wheel(0, 900); await asyncio.sleep(1.5); await pg.mouse.wheel(0, 900); await asyncio.sleep(1.5)
        await pg.mouse.wheel(0, -1800); await asyncio.sleep(2.5)
        mark("end")
        await ctx.close(); await b.close()
        path = str(await pg.video.path())
        json.dump({"video": path, "sid": sid, "marks": marks}, open(f"{S}/vid/marks.json", "w"))
        print(json.dumps({"video": path, "sid": sid, "marks": marks}))
asyncio.run(main())
