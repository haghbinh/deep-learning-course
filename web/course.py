"""Course details shared by the website hook (hooks.py) and the PDF maker
(make_pdfs.py). Change them here only."""

import re
from pathlib import Path

COURSE_TITLE = "یادگیری عمیق"
COURSE_SUBTITLE = "جزوه‌های درس"
INSTRUCTOR = "دکتر حسین حق‌بین"
UNIVERSITY = "دانشگاه خلیج فارس"
BOOK = "بر پایهٔ کتاب Understanding Deep Learning نوشتهٔ Simon J. D. Prince"
SITE_URL = "haghbinh.github.io/deep-learning-course"


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
