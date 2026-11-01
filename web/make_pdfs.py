"""Make PDFs of the lecture notes from the built site.

Run after `mkdocs build` (see web/build.sh and .github/workflows/pages.yml):

    python web/make_pdfs.py site

For every session/project page in the nav it writes site/pdf/<Sec-N>.pdf, and
one combined file site/pdf/all-sessions.pdf (cover page + all sessions in nav
order, with bookmarks). Pages are printed by Chromium, so formulas (MathJax),
callouts, images and diagrams look the same as on the website.

--offline DIR is only for testing without internet: it serves MathJax,
mermaid and the Vazirmatn font from local copies in DIR.
"""

import argparse
import functools
import html
import http.server
import re
import threading
from pathlib import Path
from urllib.parse import quote

import pikepdf
from mkdocs.config import load_config
from playwright.sync_api import sync_playwright

ALL_PDF = "all-sessions.pdf"
COMBINED_HTML = "Chapter 10/_all_sessions.html"  # same folder depth as every session page

COURSE_TITLE = "یادگیری عمیق"
COURSE_SUBTITLE = "جزوه‌های درس"
BOOK = "بر پایهٔ کتاب Understanding Deep Learning نوشتهٔ Simon J. D. Prince"

FOOTER = (
    '<div style="width:100%;font-size:8px;color:#888;text-align:center;'
    'font-family:sans-serif;"><span class="pageNumber"></span> / '
    '<span class="totalPages"></span></div>'
)

READY_JS = """() => {
  const math = [...document.querySelectorAll('.arithmatex')]
      .every(e => e.querySelector('mjx-container'));
  // Material renders each diagram into a closed shadow root; a drawn one has a height
  const mermaid = [...document.querySelectorAll('.mermaid')]
      .every(e => e.tagName === 'DIV' && e.getBoundingClientRect().height > 10);
  const imgs = [...document.images].every(i => i.complete);
  return math && mermaid && imgs;
}"""


HEADINGS_JS = r"""() => [...document.querySelectorAll('.md-content h1, .md-content h2, .md-content h3, .md-content h4, .md-content h5, .md-content h6')]
  .filter(h => h.offsetParent !== null || getComputedStyle(h).display !== 'none')
  .map(h => h.textContent.replace(/\u00b6/g, '').replace(/\s+/g, ' ').trim())"""


def pdf_name(src_uri):
    """'Chapter 12/Sec 8.md' -> 'Sec-8.pdf' (same rule as web/hooks.py)."""
    return src_uri.rsplit("/", 1)[-1][:-3].replace(" ", "-") + ".pdf"


def readme_topics(readme="README.md"):
    """{'Chapter 10/Sec 1.md': 'topic', ...} from the sessions table in README.md."""
    topics = {}
    try:
        lines = Path(readme).read_text(encoding="utf-8").splitlines()
    except OSError:
        return topics
    for line in lines:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 4:
            continue
        for m in re.finditer(r"(Chapter%20\d+/Sec%20[\w]+)\.(?:html|md)", line):
            src = m.group(1).replace("%20", " ") + ".md"
            if "_hw" not in src:
                topics[src] = cells[2]
    return topics


def session_pages(config_file):
    """[(nav label, 'Chapter 10/Sec 1.md'), ...] in nav order."""
    cfg = load_config(config_file)
    pages = []

    def walk(items):
        for item in items:
            for label, value in item.items():
                if isinstance(value, list):
                    walk(value)
                elif value.startswith("Chapter "):
                    pages.append((label, value))

    walk(cfg["nav"])
    return pages


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def serve(site_dir):
    handler = functools.partial(QuietHandler, directory=str(site_dir))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}/"


def offline_router(assets):
    assets = Path(assets)
    font_css = (
        "@font-face{font-family:'Vazirmatn';font-weight:100 900;"
        "src:url(https://fonts.gstatic.com/vazirmatn.woff2) format('woff2');}"
    )

    def route(r):
        url = r.request.url
        if "cdn.jsdelivr.net/npm/mathjax" in url:
            r.fulfill(path=assets / "mathjax/es5" / url.split("/es5/")[1])
        elif "unpkg.com/mermaid" in url:
            r.fulfill(path=assets / "mermaid.min.js")
        elif "fonts.googleapis.com" in url:
            r.fulfill(body=font_css, content_type="text/css")
        elif "fonts.gstatic.com/vazirmatn" in url:
            r.fulfill(path=assets / "Vazirmatn.woff2")
        elif url.startswith("http://127.0.0.1"):
            r.continue_()
        else:
            r.abort()

    return route


def prepare(page, url):
    page.goto(url, wait_until="networkidle")
    page.evaluate("document.querySelectorAll('details').forEach(d => d.open = true)")
    page.wait_for_function(READY_JS, timeout=120_000)
    page.evaluate("document.fonts.ready")
    page.wait_for_timeout(300)


def fix_outline(path, titles):
    """Chromium writes right-to-left bookmark titles in visual (reversed) order.
    Replace them, in document order, with the real heading texts; if the
    numbers don't match, drop the bookmarks rather than show garbled ones."""
    with pikepdf.open(path, allow_overwriting_input=True) as pdf:
        with pdf.open_outline() as outline:
            items = []

            def walk(nodes):
                for node in nodes:
                    items.append(node)
                    walk(node.children)

            walk(outline.root)
            if len(items) == len(titles):
                for item, title in zip(items, titles):
                    item.title = title
                ok = True
            else:
                outline.root.clear()
                ok = False
        pdf.save(path)
    return ok


def print_pdf(page, path):
    page.pdf(
        path=str(path),
        format="A4",
        print_background=True,
        display_header_footer=True,
        header_template="<div></div>",
        footer_template=FOOTER,
        margin={"top": "14mm", "bottom": "16mm", "left": "12mm", "right": "12mm"},
        outline=True,
        tagged=True,
    )
    if not fix_outline(path, page.evaluate(HEADINGS_JS)):
        print(f"  note: bookmarks removed from {path.name} (heading count mismatch)")


def article(html_text):
    """Inner HTML of the page's <article> (the note itself)."""
    start = html_text.index("<article")
    start = html_text.index(">", start) + 1
    end = html_text.rindex("</article>")
    body = html_text[start:end]
    return re.sub(r'<div class="pdf-download".*?</div>', "", body, count=1, flags=re.S)


def page_h1(html_text):
    m = re.search(r"<h1[^>]*>(.*?)</h1>", article(html_text), re.S)
    if not m:
        return ""
    title = re.sub(r'<a class="headerlink".*?</a>|<[^>]+>', "", m.group(1), flags=re.S)
    title = html.unescape(title).replace("\u00b6", "").strip()
    return re.sub(r"^پروژه\s*\S+\s*:\s*", "", title)  # "پروژه 1: X" -> "X"


def build_combined(site_dir, pages):
    first = (site_dir / pages[0][1].replace(".md", ".html")).read_text(encoding="utf-8")
    start = first.index("<article")
    start = first.index(">", start) + 1
    end = first.rindex("</article>")

    topics = readme_topics()
    toc = []
    for i, (label, src) in enumerate(pages):
        text = (site_dir / src.replace(".md", ".html")).read_text(encoding="utf-8")
        topic = topics.get(src) or page_h1(text)
        toc.append(f'<li><a href="#session-{i}" style="color:inherit">'
                   f"<strong>{html.escape(label)}</strong> — {html.escape(topic)}</a></li>")
    toc = "".join(toc)
    cover = (
        '<section class="print-session" style="text-align:center;padding-top:18vh">'
        f'<h1 style="font-size:2.6em;margin:0;color:inherit;font-weight:700">{COURSE_TITLE}</h1>'
        f'<p style="font-size:1.5em;margin:.4em 0 2em">{COURSE_SUBTITLE}</p>'
        f"<p>{html.escape(BOOK)}</p>"
        '<ul style="list-style:none;padding:0;margin:3em auto 0;line-height:2.1;'
        f'display:inline-block;text-align:right">{toc}</ul>'
        "</section>"
    )
    sessions = []
    for i, (_, src) in enumerate(pages):
        text = (site_dir / src.replace(".md", ".html")).read_text(encoding="utf-8")
        sessions.append(f'<section class="print-session" id="session-{i}">{article(text)}</section>')

    combined = first[:start] + cover + "".join(sessions) + first[end:]
    combined = re.sub(r"<title>.*?</title>", f"<title>{COURSE_TITLE} — {COURSE_SUBTITLE}</title>",
                      combined, count=1, flags=re.S)
    out = site_dir / COMBINED_HTML
    out.write_text(combined, encoding="utf-8")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("site_dir", nargs="?", default="site")
    ap.add_argument("--config", default="mkdocs.yml")
    ap.add_argument("--offline", help="folder with local MathJax/mermaid/font (testing only)")
    args = ap.parse_args()

    site_dir = Path(args.site_dir)
    pdf_dir = site_dir / "pdf"
    pdf_dir.mkdir(exist_ok=True)
    pages = session_pages(args.config)

    httpd, base = serve(site_dir)
    combined = build_combined(site_dir, pages)
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1100, "height": 1400})
            page.emulate_media(media="print")
            if args.offline:
                page.route("**/*", offline_router(args.offline))

            for label, src in pages:
                url = base + quote(src.replace(".md", ".html"))
                prepare(page, url)
                print_pdf(page, pdf_dir / pdf_name(src))
                print(f"{label}: pdf/{pdf_name(src)}")

            prepare(page, base + quote(COMBINED_HTML))
            print_pdf(page, pdf_dir / ALL_PDF)
            print(f"all sessions: pdf/{ALL_PDF}")
            browser.close()
    finally:
        combined.unlink(missing_ok=True)
        httpd.shutdown()


if __name__ == "__main__":
    main()
