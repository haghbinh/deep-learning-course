"""MkDocs hook: adapt Obsidian-flavoured Markdown for Python-Markdown.

Only the copy used to build the website is changed; the notes themselves
are left as written in Obsidian.

Fixes:
  * inline ``$$...$$`` inside a sentence becomes ``$...$``;
  * display-math blocks (``$$`` on its own line) get blank lines around them;
  * lists that start right after a text line get the blank line
    Python-Markdown needs.
  * code fences like ```python id="x1" lose the extra attributes;
  * session pages get "download PDF" buttons (the PDFs are made by
    web/make_pdfs.py after the site is built).
Lines inside fenced code blocks are never touched. Lines inside callouts
(``> ...``) are handled with their ``>`` prefix kept.
"""

import re

FENCE = re.compile(r"^\s*(```|~~~)")
FENCE_ATTRS = re.compile(r"^((?:\s*>)*\s*(?:```|~~~)\s*[\w+-]+)\s+[^`]*$")
PREFIX = re.compile(r"^((?:\s*>)*\s?)")
LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
INLINE_DD = re.compile(r"\$\$(?=\S)(.+?)(?<=\S)\$\$")


def _split(line):
    m = PREFIX.match(line)
    prefix = m.group(1) if ">" in m.group(1) else ""
    return prefix, line[len(prefix):]


def _blank(prefix):
    return prefix.rstrip() if prefix else ""


ALL_PDF = "all-sessions.pdf"


def pdf_name(src_uri):
    """'Chapter 12/Sec 8.md' -> 'Sec-8.pdf' (same rule as make_pdfs.py)."""
    return src_uri.rsplit("/", 1)[-1][:-3].replace(" ", "-") + ".pdf"


def _pdf_buttons(page):
    src = page.file.src_uri
    if not src.startswith("Chapter "):
        return ""
    return (
        '<div class="pdf-download" markdown="0">'
        f'<a class="md-button md-button--primary" href="../pdf/{pdf_name(src)}" download>'
        "دانلود PDF این جلسه</a> "
        f'<a class="md-button" href="../pdf/{ALL_PDF}" download>PDF همهٔ جلسه‌ها</a>'
        "</div>\n\n"
    )


def on_page_markdown(markdown, page, config, files):
    markdown = _pdf_buttons(page) + markdown
    out = []
    in_fence = False
    in_math = False
    after_math = None  # prefix of a just-closed math block

    for raw in markdown.split("\n"):
        if FENCE.match(_split(raw)[1]):
            if not in_fence:
                # ```python id="x1" -> ```python (extra attributes break the fence)
                raw = FENCE_ATTRS.sub(r"\1", raw)
            in_fence = not in_fence
            out.append(raw)
            continue
        if in_fence:
            out.append(raw)
            continue

        prefix, body = _split(raw)
        stripped = body.strip()
        prev_body = _split(out[-1])[1].strip() if out else ""

        if after_math is not None:
            if stripped:
                out.append(_blank(after_math))
            after_math = None

        if in_math:
            if stripped.endswith("$$"):
                in_math = False
                out.append(prefix + stripped)
                after_math = prefix
            else:
                out.append(raw)
            continue

        n_dd = stripped.count("$$")
        if stripped.startswith("$$") and (n_dd == 1 or (n_dd == 2 and stripped.endswith("$$"))):
            if prev_body:
                out.append(_blank(prefix))
            rest = stripped[2:]
            if rest.endswith("$$") and len(rest) >= 2:
                out.append(prefix + stripped)  # $$ ... $$ on one line
                after_math = prefix
            else:
                in_math = True
                out.append(prefix + stripped)
            continue

        # inline $$..$$ inside text -> $..$
        if "$$" in body:
            body = INLINE_DD.sub(r"$\1$", body)
            raw = prefix + body

        if LIST_ITEM.match(body) and prev_body and not LIST_ITEM.match(_split(out[-1])[1]) \
                and not _split(out[-1])[1].startswith((" ", "\t")):
            out.append(_blank(prefix))

        out.append(raw)

    return "\n".join(out)
