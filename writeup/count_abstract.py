"""Enforce the abstract's load-bearing property: the first 150 words stand alone.

The submission requirement is 150 words or fewer; the NeurIPS template says 150-250. So the
abstract is written so that its first 150 words are a complete abstract in themselves, ending at a
sentence boundary, with every load-bearing claim inside them. The remainder is additive detail.

Fails loudly rather than reporting, so it can sit in a build script.

    python count_abstract.py
"""

from __future__ import annotations

import pathlib
import re
import sys

LIMIT = 150


def main() -> int:
    tex = pathlib.Path(__file__).with_name("main.tex").read_text(encoding="utf-8")
    m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S)
    if not m:
        print("no abstract found in main.tex")
        return 1

    body = re.sub(r"%.*", "", m.group(1))              # strip LaTeX comments
    body = re.sub(r"\\[a-zA-Z]+\*?", " ", body)        # strip commands
    body = re.sub(r"[{}$\\]", " ", body)
    words = body.split()

    print("abstract: %d words total" % len(words))
    if len(words) < LIMIT:
        print("under the %d-word standalone budget; nothing to check" % LIMIT)
        return 0

    head = " ".join(words[:LIMIT])
    if not head.rstrip().endswith("."):
        print("FAIL: the first %d words do not end at a sentence boundary." % LIMIT)
        print("      ...%s" % " ".join(words[max(0, LIMIT - 14):LIMIT]))
        print("      A reader who stops at the limit must still get a complete abstract.")
        return 1

    print("OK: the first %d words end at a sentence boundary and stand alone." % LIMIT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
