"""Make PDFs of the lecture notes from the built site.

Run after `mkdocs build` (see web/build.sh and .github/workflows/pages.yml):

    python web/make_pdfs.py site

For every session/project page in the nav it writes site/pdf/<Sec-N>.pdf, and
one combined file site/pdf/all-sessions.pdf (cover page + all sessions in nav
order, with bookmarks and continuous page numbers). Pages are printed by
Chromium, so formulas (MathJax), callouts, images and diagrams look the same
as on the website. Every page has a header with the course and instructor,
the chapter and the session.

Course details (title, instructor, university) live in web/course.py.

--offline DIR is only for testing without internet: it serves MathJax,
mermaid and the Vazirmatn font from local copies in DIR.
"""

import argparse
import base64
import functools
import html
import http.server
import re
import tempfile
import threading
from pathlib import Path
from urllib.parse import quote

import pikepdf
from mkdocs.config import load_config
from playwright.sync_api import sync_playwright

from course import (BOOK, COURSE_SUBTITLE, COURSE_TITLE, INSTRUCTOR, SITE_URL,
                    UNIVERSITY, readme_topics)

ALL_PDF = "all-sessions.pdf"
COVER_HTML = "_cover.html"
FONT_FILE = Path(__file__).parent / "fonts" / "Vazirmatn-Regular.woff2"
MARGIN = {"top": "18mm", "bottom": "16mm", "left": "12mm", "right": "12mm"}

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
  .map(h => h.textContent.replace(/¶/g, '').replace(/\s+/g, ' ').trim())"""


# ---------------------------------------------------------------- header/footer

def _font_face():
    """Header/footer templates can't use the page's web fonts, so the
    Vazirmatn font (web/fonts, SIL OFL) is embedded in the template."""
    if not FONT_FILE.exists():
        return "", "sans-serif"
    data = base64.b64encode(FONT_FILE.read_bytes()).decode()
    face = ("<style>@font-face{font-family:'VazirHdr';"
            f"src:url(data:font/woff2;base64,{data}) format('woff2');}}</style>")
    return face, "'VazirHdr', sans-serif"


FONT_FACE, HEADER_FONT = _font_face()

FOOTER_NUMBERS = (
    '<div style="width:100%;font-size:8px;color:#888;text-align:center;'
    'font-family:sans-serif;"><span class="pageNumber"></span> / '
    '<span class="totalPages"></span></div>'
)
FOOTER_EMPTY = "<div></div>"  # combined PDF: numbers are stamped after merging


def header(chapter, session):
    """Right: course — instructor | middle: chapter | left: session."""
    e = html.escape
    cell = "flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;"
    return (
        f"{FONT_FACE}"
        f'<div dir="rtl" style="width:100%;margin:0 12mm;padding-bottom:2px;'
        f"border-bottom:0.5px solid #bbb;display:flex;gap:10px;"
        f'font-family:{HEADER_FONT};font-size:8px;color:#555;">'
        f'<span style="{cell}text-align:right">{e(COURSE_TITLE)} — {e(INSTRUCTOR)}</span>'
        f'<span style="{cell}text-align:center">{e(chapter)}</span>'
        f'<span style="{cell}text-align:left">{e(session)}</span></div>'
    )


# ---------------------------------------------------------------- site helpers

def pdf_name(src_uri):
    """'Chapter 12/Sec 8.md' -> 'Sec-8.pdf' (same rule as web/hooks.py)."""
    return src_uri.rsplit("/", 1)[-1][:-3].replace(" ", "-") + ".pdf"


def session_pages(config_file):
    """[(chapter label, session label, 'Chapter 10/Sec 1.md'), ...] in nav order."""
    cfg = load_config(config_file)
    pages = []

    def walk(items, section):
        for item in items:
            for label, value in item.items():
                if isinstance(value, list):
                    walk(value, label)
                elif value.startswith("Chapter "):
                    pages.append((section, label, value))

    walk(cfg["nav"], "")
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


def article(html_text):
    """Inner HTML of the page's <article> (the note itself)."""
    start = html_text.index("<article")
    start = html_text.index(">", start) + 1
    end = html_text.rindex("</article>")
    return html_text[start:end]


def page_h1(html_text):
    m = re.search(r"<h1[^>]*>(.*?)</h1>", article(html_text), re.S)
    if not m:
        return ""
    title = re.sub(r'<a class="headerlink".*?</a>|<[^>]+>', "", m.group(1), flags=re.S)
    title = html.unescape(title).replace("¶", "").strip()
    return re.sub(r"^پروژه\s*\S+\s*:\s*", "", title)  # "پروژه 1: X" -> "X"


def session_topic(site_dir, src):
    """Topic from the README sessions table, else the page's main heading."""
    text = (site_dir / src.replace(".md", ".html")).read_text(encoding="utf-8")
    return readme_topics().get(src) or page_h1(text)


# ---------------------------------------------------------------- PDF helpers

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
            ok = len(items) == len(titles)
            if ok:
                for item, title in zip(items, titles):
                    item.title = title
            else:
                outline.root.clear()
        pdf.save(path)
    return ok


def print_pdf(page, path, header_html, footer_html):
    page.pdf(
        path=str(path),
        format="A4",
        print_background=True,
        display_header_footer=True,
        header_template=header_html,
        footer_template=footer_html,
        margin=MARGIN,
        outline=True,
        tagged=True,
    )
    if not fix_outline(path, page.evaluate(HEADINGS_JS)):
        print(f"  note: bookmarks removed from {path.name} (heading count mismatch)")


def _digits_width(text, size):
    """Width of a Helvetica string made of digits, spaces and '/'."""
    widths = {" ": 278, "/": 278}
    return sum(widths.get(ch, 556) for ch in text) * size / 1000


def stamp_page_numbers(pdf, first):
    """Write 'n / N' at the bottom centre of every page from index `first` on."""
    total = len(pdf.pages)
    font = pdf.make_indirect(pikepdf.Dictionary(
        Type=pikepdf.Name.Font, Subtype=pikepdf.Name.Type1,
        BaseFont=pikepdf.Name.Helvetica))
    for i in range(first, total):
        page = pdf.pages[i]
        text = f"{i + 1} / {total}"
        x0, _, x1, _ = [float(v) for v in page.mediabox]
        x = (x0 + x1) / 2 - _digits_width(text, 8) / 2
        if "/Resources" not in page.obj:
            page.obj.Resources = pikepdf.Dictionary()
        resources = page.obj.Resources
        if "/Font" not in resources:
            resources.Font = pikepdf.Dictionary()
        resources.Font.NumF = font
        # wrap Chromium's drawing in q/Q so its coordinate changes don't move the number
        page.contents_add(pdf.make_stream(b"q\n"), prepend=True)
        stream = f"\nQ q BT 0.53 0.53 0.53 rg /NumF 8 Tf {x:.2f} 22 Td ({text}) Tj ET Q".encode()
        page.contents_add(pdf.make_stream(stream), prepend=False)


# ---------------------------------------------------------------- combined PDF

def write_cover(site_dir, entries):
    """entries: [(label, topic, first page number)]. Uses the site's page shell."""
    shell = (site_dir / "index.html").read_text(encoding="utf-8")
    start = shell.index("<article")
    start = shell.index(">", start) + 1
    end = shell.rindex("</article>")
    e = html.escape
    rows = "".join(
        "<tr>"
        f'<td style="padding:3px 0;white-space:nowrap;border:none"><strong>{e(label)}</strong></td>'
        f'<td style="padding:3px 14px;border:none">{e(topic)}</td>'
        f'<td style="padding:3px 0;text-align:left;border:none">{num}</td>'
        "</tr>"
        for label, topic, num in entries
    )
    cover = (
        '<div style="text-align:center;padding-top:8vh">'
        f'<h1 style="font-size:2.6em;margin:0;color:inherit;font-weight:700">{e(COURSE_TITLE)}</h1>'
        f'<p style="font-size:1.5em;margin:.3em 0 1.6em">{e(COURSE_SUBTITLE)}</p>'
        f'<p style="font-size:1.25em;margin:0">مدرس: <strong>{e(INSTRUCTOR)}</strong></p>'
        f'<p style="font-size:1.1em;margin:.3em 0 1.6em">{e(UNIVERSITY)}</p>'
        f'<p style="margin:0 0 2.4em">{e(BOOK)}</p>'
        '<table style="margin:0 auto;border-collapse:collapse;font-size:.95em;'
        f'display:table;width:auto;border:none;box-shadow:none">{rows}</table>'
        f'<p style="margin-top:2.4em;color:#888;font-size:.85em">{e(SITE_URL)}</p>'
        "</div>"
    )
    out = site_dir / COVER_HTML
    out.write_text(shell[:start] + cover + shell[end:], encoding="utf-8")
    return out


def merge(cover_pdf, parts, out_path):
    """parts: [(bookmark title, part pdf path)] -> one PDF with bookmarks and
    page numbers (the cover is page 1 and carries no number)."""
    merged = pikepdf.new()
    opened = []
    with pikepdf.open(cover_pdf) as cover:
        merged.pages.extend(cover.pages)
    n_cover = len(merged.pages)
    toc = []  # (title, start index, [(title, page index, top, level), ...])
    for title, path in parts:
        part = pikepdf.open(path)
        opened.append(part)
        start = len(merged.pages)
        kids = []
        with part.open_outline() as outline:
            def walk(nodes, level):
                for node in nodes:
                    dest = node.destination
                    if isinstance(dest, pikepdf.Array) and len(dest) >= 4:
                        idx = part.pages.index(pikepdf.Page(dest[0]))
                        try:
                            top = float(dest[3])
                        except (TypeError, ValueError):
                            top = None
                        kids.append((node.title, start + idx, top, level))
                    walk(node.children, level + 1)
            walk(outline.root, 0)
        merged.pages.extend(part.pages)
        toc.append((title, start, kids))

    with merged.open_outline() as outline:
        outline.root.append(pikepdf.OutlineItem(COURSE_TITLE, 0))
        for title, start, kids in toc:
            top_item = pikepdf.OutlineItem(title, start)
            stack = [(-1, top_item)]
            for kid_title, idx, top, level in kids:
                if top is None:
                    item = pikepdf.OutlineItem(kid_title, idx)
                else:
                    item = pikepdf.OutlineItem(kid_title, idx, "XYZ", top=top)
                while stack[-1][0] >= level:
                    stack.pop()
                stack[-1][1].children.append(item)
                stack.append((level, item))
            outline.root.append(top_item)

    stamp_page_numbers(merged, n_cover)
    merged.save(out_path)
    for part in opened:
        part.close()


# ---------------------------------------------------------------- main

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
    tmp = Path(tempfile.mkdtemp(prefix="pdfparts-"))

    httpd, base = serve(site_dir)
    cover_html = None
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1100, "height": 1400})
            page.emulate_media(media="print")
            if args.offline:
                page.route("**/*", offline_router(args.offline))

            parts = []
            for i, (chapter, label, src) in enumerate(pages):
                topic = session_topic(site_dir, src)
                session = f"{label}: {topic}" if topic else label
                hdr = header(chapter, session)
                prepare(page, base + quote(src.replace(".md", ".html")))
                print_pdf(page, pdf_dir / pdf_name(src), hdr, FOOTER_NUMBERS)
                part = tmp / f"part-{i:02d}.pdf"
                print_pdf(page, part, hdr, FOOTER_EMPTY)
                parts.append((session, topic, label, part))
                print(f"{label}: pdf/{pdf_name(src)}")

            # the cover lists the page where each session starts
            sizes = []
            for _, _, _, part in parts:
                with pikepdf.open(part) as pdf:
                    sizes.append(len(pdf.pages))
            cover_pdf = tmp / "cover.pdf"
            n_cover = 1
            for _ in range(3):  # redo if the cover itself runs past one page
                entries, start = [], n_cover + 1
                for (_, topic, label, _), size in zip(parts, sizes):
                    entries.append((label, topic, start))
                    start += size
                cover_html = write_cover(site_dir, entries)
                prepare(page, base + quote(COVER_HTML))
                page.pdf(path=str(cover_pdf), format="A4", print_background=True,
                         margin=MARGIN, tagged=True)
                with pikepdf.open(cover_pdf) as pdf:
                    if len(pdf.pages) == n_cover:
                        break
                    n_cover = len(pdf.pages)
            browser.close()

        merge(cover_pdf, [(session, part) for session, _, _, part in parts],
              pdf_dir / ALL_PDF)
        print(f"all sessions: pdf/{ALL_PDF}")
    finally:
        if cover_html:
            cover_html.unlink(missing_ok=True)
        httpd.shutdown()


if __name__ == "__main__":
    main()
