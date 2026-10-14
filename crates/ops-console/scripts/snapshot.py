#!/usr/bin/env python3
"""Screenshot rendered console pages with headless Chrome and check that none
scrolls sideways.

Reads the `<stem>.<theme>.html` files `scripts/snapshot.sh` rendered into DIR,
serves DIR on a loopback port (the pages link `/static/console.css`), and for
each page and width writes a full-page PNG to DIR/png. It also measures the
page: a document wider than the viewport scrolls sideways, which a table may
only do inside its own scroll box. Every such page is listed with the
elements that stick out, and the exit status is 1.

Then one contact sheet per page goes to DIR/sheets: the phone and desktop
widths, light and dark, side by side.

Chrome is driven over the DevTools protocol on a pipe (fds 3 and 4), so this
needs nothing beyond Python's standard library and a Chrome binary.
"""

import base64
import functools
import html
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# (width, height, mobile): a phone, a tablet, a laptop.
VIEWPORTS = [(390, 844, True), (768, 1024, True), (1280, 800, False)]
THEMES = ["light", "dark"]
# Contact-sheet panels: (width, theme, scale).
SHEET = [(390, "light", 1), (390, "dark", 1), (1280, "light", 0.5), (1280, "dark", 0.5)]

# The innermost elements whose box or content reaches past the viewport with
# no scroll box around them: the cause of a page that scrolls sideways. An
# element's content counts too, since a long unbroken word overflows its box
# without widening it.
MEASURE = """(() => {
  const de = document.documentElement, cw = de.clientWidth;
  const scrolls = el => ['auto', 'scroll', 'hidden', 'clip'].includes(getComputedStyle(el).overflowX);
  const boxed = el => {
    for (let p = el.parentElement; p; p = p.parentElement) if (scrolls(p)) return true;
    return false;
  };
  const extent = el => {
    const r = el.getBoundingClientRect();
    return scrolls(el) ? r.right : Math.max(r.right, r.left + el.scrollWidth);
  };
  const over = el => extent(el) > cw + 0.5 && !boxed(el);
  const wide = [...document.body.querySelectorAll('*')]
    .filter(el => over(el) && ![...el.children].some(over))
    .slice(0, 5)
    .map(el => el.tagName.toLowerCase() + ' ' + (el.getAttribute('class') || '').slice(0, 80));
  return JSON.stringify({scrollWidth: de.scrollWidth, clientWidth: cw, wide});
})()"""


class Chrome:
    """Headless Chrome, spoken to over `--remote-debugging-pipe`: JSON messages
    separated by NUL, commands on fd 3, replies and events on fd 4."""

    def __init__(self, binary):
        to_chrome, self._send = os.pipe()
        self._recv, from_chrome = os.pipe()

        def wire():
            # Fresh copies first, so neither dup2 can overwrite the other's source.
            r, w = os.dup(to_chrome), os.dup(from_chrome)
            os.dup2(r, 3)
            os.dup2(w, 4)
            os.set_inheritable(3, True)
            os.set_inheritable(4, True)

        self._profile = tempfile.mkdtemp(prefix="snapshot-chrome-")
        args = [
            binary,
            "--headless",
            "--remote-debugging-pipe",
            f"--user-data-dir={self._profile}",
            "--no-first-run",
            "--no-default-browser-check",
            "--hide-scrollbars",
            "--disable-gpu",
            "about:blank",
        ]
        self._proc = None
        self._buf = b""
        self._next_id = 0
        self._loads = 0
        try:
            # close_fds=False: Python opens every fd non-inheritable, so only
            # 3 and 4, made inheritable above, reach Chrome.
            self._proc = subprocess.Popen(
                args,
                close_fds=False,
                preexec_fn=wire,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            os.close(to_chrome)
            os.close(from_chrome)
            target = self.call("Target.createTarget", url="about:blank")["targetId"]
            self._session = self.call(
                "Target.attachToTarget", targetId=target, flatten=True
            )["sessionId"]
        except BaseException:
            # The caller gets no object to close, so stop Chrome and remove
            # its profile here.
            if self._proc:
                self._proc.kill()
                self._proc.wait()
            shutil.rmtree(self._profile, ignore_errors=True)
            raise

    def call(self, method, session=None, **params):
        self._next_id += 1
        msg = {"id": self._next_id, "method": method, "params": params}
        if session:
            msg["sessionId"] = session
        os.write(self._send, json.dumps(msg).encode() + b"\0")
        while True:
            reply = self._read()
            if reply.get("id") == self._next_id:
                if "error" in reply:
                    raise RuntimeError(f"{method}: {reply['error']}")
                return reply["result"]

    def _read(self):
        while b"\0" not in self._buf:
            chunk = os.read(self._recv, 1 << 20)
            if not chunk:
                raise RuntimeError("Chrome closed the DevTools pipe")
            self._buf += chunk
        msg, self._buf = self._buf.split(b"\0", 1)
        return json.loads(msg)

    def page(self, method, **params):
        return self.call(method, session=self._session, **params)

    def shoot(self, url, width, height, mobile, scheme):
        """Load `url` at this viewport, with the OS colour scheme `scheme`;
        return (PNG bytes, measurement)."""
        self.page(
            "Emulation.setDeviceMetricsOverride",
            width=width,
            height=height,
            deviceScaleFactor=1,
            mobile=mobile,
        )
        # A page with no saved theme (the error page) follows the OS, and
        # headless Chrome's OS is light: a dark shot must say dark.
        self.page(
            "Emulation.setEmulatedMedia",
            features=[{"name": "prefers-color-scheme", "value": scheme}],
        )
        # A query unique to this load, so the document it waits for is never
        # the one already open: the same page is loaded once per width.
        self._loads += 1
        url = f"{url}?load={self._loads}"
        self.page("Page.navigate", url=url)
        # Until the new document has loaded, stylesheet included.
        loaded = (
            f"location.href === {json.dumps(url)} && document.readyState === 'complete'"
        )
        deadline = time.monotonic() + 30
        while not self.page("Runtime.evaluate", expression=loaded)["result"]["value"]:
            if time.monotonic() > deadline:
                raise RuntimeError(f"{url} did not load within 30s")
            time.sleep(0.02)
        measured = json.loads(
            self.page("Runtime.evaluate", expression=MEASURE)["result"]["value"]
        )
        # The whole document, and as wide as it really is, so a page that
        # scrolls sideways shows it.
        size = self.page("Page.getLayoutMetrics")["cssContentSize"]
        clip = {
            "x": 0,
            "y": 0,
            "width": max(size["width"], width),
            "height": size["height"],
            "scale": 1,
        }
        shot = self.page(
            "Page.captureScreenshot",
            format="png",
            captureBeyondViewport=True,
            clip=clip,
        )
        return base64.b64decode(shot["data"]), measured

    def close(self):
        try:
            self.call("Browser.close")
        except RuntimeError as e:
            # Chrome may exit before it answers; anything else is news.
            if "closed the DevTools pipe" not in str(e):
                print(f"Browser.close: {e}", file=sys.stderr)
        try:
            self._proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            print("Chrome did not exit within 10s; killing it", file=sys.stderr)
            self._proc.kill()
            self._proc.wait()
        shutil.rmtree(self._profile, ignore_errors=True)


def serve(root):
    """Serve `root` on a free loopback port, quietly; return its origin."""

    class Quiet(SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(Quiet, directory=str(root))
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_address[1]}"


def find_chrome():
    for name in [
        os.environ.get("CHROME"),
        "google-chrome",
        "google-chrome-stable",
        "chromium",
        "chromium-browser",
    ]:
        if name and shutil.which(name):
            return shutil.which(name)
    sys.exit("no Chrome found: set CHROME to its binary")


def main():
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} DIR")
    root = Path(sys.argv[1]).resolve()
    try:
        stems = (root / "pages.txt").read_text().split()
    except OSError as e:
        sys.exit(
            f"cannot read the page list: {e}. Run scripts/snapshot.sh, which writes it."
        )
    pngs, sheets = root / "png", root / "sheets"
    pngs.mkdir(exist_ok=True)
    sheets.mkdir(exist_ok=True)
    origin = serve(root)
    chrome = Chrome(find_chrome())
    sideways = []
    try:
        for stem in stems:
            for theme in THEMES:
                for width, height, mobile in VIEWPORTS:
                    png, m = chrome.shoot(
                        f"{origin}/{stem}.{theme}.html", width, height, mobile, theme
                    )
                    (pngs / f"{stem}.{theme}.{width}.png").write_bytes(png)
                    if m["scrollWidth"] > m["clientWidth"]:
                        sideways.append(
                            f"{stem}.{theme} at {width}px: {m['scrollWidth']}px wide; {m['wide']}"
                        )
            panels = "".join(
                f'<figure><figcaption>{w}px {t}</figcaption><img src="/png/{html.escape(stem)}.{t}.{w}.png" width="{int(w * s)}"></figure>'
                for w, t, s in SHEET
            )
            (sheets / f"{stem}.html").write_text(
                "<!DOCTYPE html><html><head><meta charset='utf-8'><link rel='stylesheet' href='sheet.css'></head>"
                f"<body><h1>{html.escape(stem)}</h1><div>{panels}</div></body></html>"
            )
        (sheets / "sheet.css").write_text(
            "body{margin:16px;font:14px system-ui,sans-serif;background:#888}"
            "div{display:flex;gap:16px;align-items:flex-start}"
            "figure{margin:0}img{display:block;outline:1px solid #444}h1{font-size:18px;margin:0 0 8px}"
        )
        sheet_width = int(sum(w * s for w, _, s in SHEET)) + 16 * (len(SHEET) + 1)
        for stem in stems:
            png, _ = chrome.shoot(
                f"{origin}/sheets/{stem}.html", sheet_width, 800, False, "light"
            )
            (sheets / f"{stem}.png").write_bytes(png)
    finally:
        chrome.close()

    shots = len(stems) * len(THEMES) * len(VIEWPORTS)
    print(f"{shots} screenshots in {pngs}, {len(stems)} contact sheets in {sheets}")
    if sideways:
        print("Pages that scroll sideways:", *sideways, sep="\n  ")
        sys.exit(1)
    print("No page scrolls sideways at any width.")


if __name__ == "__main__":
    main()
