"""
rag/pdf_extract.py — table-aware extraction of the WHO guideline PDFs.

Turns a PDF into a list of `Piece`s: self-contained pieces of guideline text,
each carrying its section heading. `rag/build_kb.py` chunks, checks and
embeds them.

Three things make these PDFs hard, and each has a dedicated step here:

1. GLYPH-ID TEXT. Some embedded fonts (the IMCI chart booklet's headings, parts
   of WHO PEN) have no Unicode map, so PyMuPDF returns glyph ids instead of
   characters. For these TrueType fonts glyph id = ASCII code - 29, and the
   space is glyph 3 ("\\x03"). `repair_glyph_text()` reverses that, plus the
   ligature and symbol glyphs seen in these files (fi, ff, fl, >=).

2. CLASSIFICATION TABLES (IMCI chart booklet). Each classification row is drawn
   as three filled cells: signs | classification | treatment, coloured pink
   (urgent referral), yellow (specific treatment) or green (home care).
   PyMuPDF's find_tables() finds the grid but splits and merges these rows and
   drops the spaces between words, so the rows are taken from the coloured
   cells themselves and the text of each cell is read by clipping the page to
   that cell. Each row becomes one readable piece:
   "Signs: ... -> Classification: ... (Pink: urgent referral) -> Treatment: ...".

3. OTHER TABLES (dose tables, PEN protocols). find_tables() gives the cell
   grid; cell text is read by clipping (so spaces survive) and each row becomes
   "header: value; header: value".

Text outside tables is kept as prose under the current heading.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

# ─── Glyph-id repair ─────────────────────────────────────────────────────────

GLYPH_OFFSET = 29
# Non-ASCII glyphs found inside glyph-id spans of these three documents, and
# what they stand for in context (audited span by span; e.g. "individualȇs",
# "100 Ɖg twice daily", "37.5\x83C", "Diɝcult breathing", "ACEΖ"). Applied
# only to spans that are being glyph-decoded.
GLYPH_SPECIALS = {
    "ȴ": "fi", "΍": "ff", "ȵ": "fl", "ɝ": "ffi",
    "Ȳ": "≥", "ȱ": "≤",
    "Ȃ": "–", "²": "–",
    "ȇ": "'", "¶": "'", "Ȋ": "“", "ȋ": "”",
    "Ζ": "I",
    "Ɖ": "µ", "\x97": "µ",
    "\x83": "°",
}
# Characters glyph-shifted capitals are made of ('$' = 'A', '&' = 'C', ',' = 'I', ...)
_SHIFT_MARKS = set("$&'()*+,/<>[\\]^")


def _looks_glyph_encoded(text: str) -> bool:
    """A span is glyph-encoded if it contains the glyph-3 space or other C0 controls."""
    return any(ord(c) < 32 and c not in "\t\n\r" for c in text)


def _alpha_ratio(text: str) -> float:
    s = text.replace(" ", "")
    return sum(c.isalpha() for c in s) / len(s) if s else 0.0


def decode_glyphs(text: str) -> str:
    out = []
    for c in text:
        if c in GLYPH_SPECIALS:
            out.append(GLYPH_SPECIALS[c])
        elif ord(c) < 128 and 32 <= ord(c) + GLYPH_OFFSET < 127:
            out.append(chr(ord(c) + GLYPH_OFFSET))
        else:
            out.append(c)
    return "".join(out)


def repair_glyph_text(text: str, font_is_glyph_encoded: bool = False) -> str:
    """
    Decode a span if it is glyph-encoded.

    A span is decoded when it contains glyph-3 spaces / control characters, or
    when its font is glyph-encoded somewhere in this document AND the span
    looks like one shifted capital word: no ASCII letters or spaces, 3+
    characters, at least one shift mark,
    and the decoded text is mostly letters (',1)(&7,21' -> 'INFECTION').
    The same fonts also carry correctly encoded digits, ISBNs and URLs in WHO
    PEN ('2020', '(PEN)'), which must be left alone.
    """
    if _looks_glyph_encoded(text):
        return decode_glyphs(text)
    if font_is_glyph_encoded and text.strip():
        has_letters = any("a" <= c.lower() <= "z" for c in text)
        word = text.strip()
        if (not has_letters and len(word) >= 3 and not any(c.isspace() for c in word)
                and any(c in _SHIFT_MARKS for c in word)):
            decoded = decode_glyphs(text)
            if _alpha_ratio(decoded) >= 0.7:
                return decoded
    return text


def glyph_encoded_fonts(doc) -> set[str]:
    """Fonts that produce glyph-3 spaces anywhere in the document."""
    fonts: set[str] = set()
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    if _looks_glyph_encoded(span["text"]):
                        fonts.add(span["font"])
    return fonts


# ─── Lines with geometry ─────────────────────────────────────────────────────

@dataclass
class Line:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    size: float
    bold: bool

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def page_lines(page, bad_fonts: set[str], clip=None) -> list[Line]:
    """Repaired text lines of a page (optionally clipped), in PyMuPDF reading order."""
    lines: list[Line] = []
    d = page.get_text("dict", clip=clip) if clip is not None else page.get_text("dict")
    for block in d["blocks"]:
        for line in block.get("lines", []):
            spans = line["spans"]
            if not spans:
                continue
            text = "".join(repair_glyph_text(s["text"], s["font"] in bad_fonts) for s in spans)
            text = _clean(text)
            if not text:
                continue
            x0, y0, x1, y1 = line["bbox"]
            lines.append(Line(
                text=text, x0=x0, y0=y0, x1=x1, y1=y1,
                size=max(s["size"] for s in spans),
                bold=any((s["flags"] & 16) or "Bold" in s["font"] for s in spans),
            ))
    return lines


def clip_text(page, rect, bad_fonts: set[str]) -> str:
    """Text inside a rectangle, repaired and with words joined by spaces."""
    return _clean(" ".join(l.text for l in page_lines(page, bad_fonts, clip=rect)))


# ─── Pieces ──────────────────────────────────────────────────────────────────

@dataclass
class Piece:
    """A self-contained piece of guideline text with its heading."""
    text: str
    section: str
    page_number: int
    kind: str  # classification_row | table_row | text
    meta: dict = field(default_factory=dict)


# Colour of a classification cell -> what the colour means in IMCI
IMCI_COLOURS = {
    "pink": "Pink: urgent pre-referral treatment and referral",
    "yellow": "Yellow: specific medical treatment and advice",
    "green": "Green: simple advice on home management",
}


def _colour_name(fill) -> str | None:
    if not fill or len(fill) < 3:
        return None
    r, g, b = fill[:3]
    if r > 0.9 and g < 0.85 and b < 0.85:
        return "pink"
    if r > 0.9 and g > 0.9 and b < 0.8:
        return "yellow"
    if g > 0.9 and r < 0.85 and b < 0.85:
        return "green"
    return None


def classification_rows(page) -> list[dict]:
    """
    Coloured classification rows on a chart-booklet page.

    Returns [{"colour", "y0", "y1", "cells": [Rect, ...] sorted by x}], one per
    row. Cells smaller than 30 x 10 pt are ignored (bullets, swatches).
    """
    import pymupdf

    by_row: dict[tuple, list] = {}
    for d in page.get_drawings():
        colour = _colour_name(d.get("fill"))
        r = d["rect"]
        if colour is None or r.width < 30 or r.height < 10:
            continue
        key = (colour, round(r.y0), round(r.y1))
        by_row.setdefault(key, []).append(pymupdf.Rect(r))
    rows = []
    for (colour, y0, y1), cells in by_row.items():
        cells.sort(key=lambda c: c.x0)
        rows.append({"colour": colour, "y0": y0, "y1": y1, "cells": cells})
    rows.sort(key=lambda r: (r["y0"], r["cells"][0].x0))
    return rows


def _row_groups(rows: list[dict], gap: float = 4.0) -> list[list[dict]]:
    """Group vertically contiguous rows into one classification table."""
    groups: list[list[dict]] = []
    for row in rows:
        if groups and abs(row["y0"] - groups[-1][-1]["y1"]) <= gap \
                and abs(row["cells"][0].x0 - groups[-1][-1]["cells"][0].x0) < 5:
            groups[-1].append(row)
        else:
            groups.append([row])
    return groups


# ─── Headings ────────────────────────────────────────────────────────────────

_NOISE = re.compile(
    r"^(page\s*\d*\s*(of\s*\d+)?|\d{1,3}|[ivxlc]{1,6}|▼.*|.*▼\s*[ivxlc\d]+|.*IMCI HANDBOOK.*)$",
    re.I,
)


def is_noise(line: Line, page_height: float) -> bool:
    """Page numbers, running headers/footers."""
    if line.y0 > page_height * 0.955:
        return True
    return bool(_NOISE.match(line.text.strip()))


def body_font_size(lines: Iterable[Line]) -> float:
    from collections import Counter
    c = Counter()
    for l in lines:
        c[round(l.size, 1)] += len(l.text)
    return c.most_common(1)[0][0] if c else 10.0


def is_heading(line: Line, body: float) -> bool:
    t = line.text.strip()
    if len(t) < 4 or len(t) > 110 or t.endswith((",", ";")):
        return False
    letters = [c for c in t if c.isalpha()]
    if len(letters) < 4:
        return False
    upper = sum(c.isupper() for c in letters) / len(letters)
    if line.size >= body * 1.25:
        return True
    if line.bold and upper > 0.8:
        return True
    if re.match(r"^(CHAPTER|ANNEX|MODULE|PROTOCOL)\s+[\dA-Z]", t):
        return True
    return re.match(r"^\d+(\.\d+)+\s+[A-Z]", t) is not None and line.bold


# ─── Fragment check for clipped cell text ────────────────────────────────────

def page_vocabulary(page, bad_fonts: set[str]) -> set[str]:
    """Lower-cased words of the whole page, read line by line (never clipped)."""
    words: set[str] = set()
    for l in page_lines(page, bad_fonts):
        words.update(w.lower() for w in re.findall(r"[A-Za-z]{2,}", l.text))
    return words


def fragment_ratio(text: str, vocabulary: set[str]) -> float:
    """
    Share of words in clipped cell text that do not occur as whole words on the
    page. Clipping a cell whose text runs across cell borders (rotated headers,
    multi-line cells) cuts words ('tchy', 'minal'); clean cells score ~0.
    """
    tokens = [w.lower() for w in re.findall(r"[A-Za-z]{2,}", text)]
    if not tokens:
        return 0.0
    return sum(t not in vocabulary for t in tokens) / len(tokens)


# ─── Generic table rows ──────────────────────────────────────────────────────

def is_line_split(rows: list[list[str]]) -> bool:
    """
    True when find_tables() returned one row per text LINE of multi-line cells
    ("SIGNS: days and; UNIQUE FEATURES IN HIV: Complications more frequent").
    Sign: a sparse key column (e.g. CLASSIFY AS, filled once per logical row,
    often vertically centred). Merging such rows back was tried and mixed
    neighbouring conditions, so these tables are read as prose instead.
    Dose tables, where every column is filled on most rows, are not affected.
    """
    if len(rows) < 4:
        return False
    ncols = len(rows[0])
    counts = [sum(bool(r[i]) for r in rows) for i in range(ncols)]
    candidates = [c for c in counts if c >= 2]
    return bool(candidates) and min(candidates) <= 0.6 * len(rows)


def table_pieces(page, bad_fonts: set[str], section: str, min_cols: int = 2):
    """
    Rows of find_tables() tables as 'header: value; ...' text.

    Returns (pieces, table_rects). Tables with fewer than `min_cols` columns or
    a single row are not treated as tables (their text stays prose).
    """
    import pymupdf

    pieces, rects = [], []
    vocab = None
    for t in page.find_tables().tables:
        if t.col_count < min_cols or t.row_count < 2:
            continue
        if vocab is None:
            vocab = page_vocabulary(page, bad_fonts)
        grid = []
        for row in t.rows:
            grid.append([clip_text(page, pymupdf.Rect(c), bad_fonts) if c else "" for c in row.cells])
        header = grid[0]
        if sum(bool(h) for h in header) < 2:
            header = [""] * len(header)
            body_rows = grid
        else:
            body_rows = grid[1:]
        if is_line_split(body_rows):
            continue  # leave the region to prose extraction
        rects.append(pymupdf.Rect(t.bbox))
        for cells in body_rows:
            parts = []
            for h, v in zip(header, cells):
                if not v:
                    continue
                parts.append(f"{h}: {v}" if h and h != v else v)
            text = "; ".join(parts)
            if len(text) >= 20:
                pieces.append(Piece(text=text, section=section, page_number=page.number + 1, kind="table_row",
                                    meta={"fragment_ratio": round(fragment_ratio(text, vocab), 3)}))
    return pieces, rects


# ─── Document extractors ─────────────────────────────────────────────────────

def _inside(line: Line, rects) -> bool:
    return any(r.x0 - 1 <= line.cx <= r.x1 + 1 and r.y0 - 1 <= line.cy <= r.y1 + 1 for r in rects)


def _prose_pieces(lines: list[Line], page_no: int, section_of) -> list[Piece]:
    """Group consecutive lines that share a section into prose pieces."""
    pieces: list[Piece] = []
    buf: list[str] = []
    current = None
    for line in lines:
        sec = section_of(line)
        if sec != current and buf:
            pieces.append(Piece(text=" ".join(buf), section=current, page_number=page_no, kind="text"))
            buf = []
        current = sec
        if not section_of.is_heading_line(line):
            buf.append(line.text)
    if buf:
        pieces.append(Piece(text=" ".join(buf), section=current, page_number=page_no, kind="text"))
    return pieces


class _SectionTracker:
    """Keeps 'Document title > heading' across pages."""

    def __init__(self, title: str, body_size: float):
        self.title = title
        self.body = body_size
        self.heading = ""
        self._heading_lines: set[int] = set()

    def is_heading_line(self, line: Line) -> bool:
        return id(line) in self._heading_lines

    def __call__(self, line: Line) -> str:
        if is_heading(line, self.body):
            if id(line) not in self._heading_lines:
                self._heading_lines.add(id(line))
                # Consecutive heading lines (wrapped titles) are joined
                if self._last_was_heading and len(self.heading) + len(line.text) < 140:
                    self.heading = f"{self.heading} {line.text}"
                else:
                    self.heading = line.text
            self._last_was_heading = True
        else:
            self._last_was_heading = False
        return f"{self.title} > {self.heading}" if self.heading else self.title

    _last_was_heading = False


def extract_imci_chart_booklet(doc, title: str, pages: Iterable[int]) -> list[Piece]:
    """IMCI chart booklet: coloured classification rows + dose tables + prose."""
    import pymupdf

    bad_fonts = glyph_encoded_fonts(doc)
    out: list[Piece] = []
    for pno in pages:
        page = doc[pno - 1]
        H = page.rect.height
        lines = [l for l in page_lines(page, bad_fonts) if not is_noise(l, H)]
        if not lines:
            continue
        body = body_font_size(lines)
        rows = classification_rows(page)
        cell_rects = [c for r in rows for c in r["cells"]]
        vocab = page_vocabulary(page, bad_fonts) if rows else set()
        # Page title: the topmost heading-like line outside the coloured cells
        top = sorted((l for l in lines if not _inside(l, cell_rects)), key=lambda l: l.y0) or lines
        page_title = next((l.text for l in top if is_heading(l, body) and l.y0 < H * 0.2), top[0].text)
        page_title = page_title.rstrip(":")
        groups = _row_groups(rows)
        band_rects = []
        for group in groups:
            gy0, gy1 = group[0]["y0"], group[-1]["y1"]
            x_signs = group[0]["cells"][0].x0
            band_rects.append(pymupdf.Rect(x_signs, gy0, page.rect.x1, gy1))
            # The symptom question this table answers ("Does the child have
            # diarrhoea?"): the nearest bold question, or "Does the ..." line,
            # above it. The rotated "Classify ..." side labels extract as
            # fragments, so they are not used.
            questions = [l for l in lines if l.y1 <= gy0 + 2 and l.text.endswith("?")
                         and (l.bold or l.text.lower().startswith(("does ", "is the ", "has the ")))]
            question = max(questions, key=lambda l: l.y1).text if questions else ""
            section = " > ".join(p for p in (title, page_title, question) if p)
            for row in group:
                cells = row["cells"]
                texts = [clip_text(page, c, bad_fonts) for c in cells]
                signs = texts[0] if len(texts) >= 3 else ""
                classification = texts[1] if len(texts) >= 3 else texts[0]
                treatment = " ".join(texts[2:]) if len(texts) >= 3 else " ".join(texts[1:])
                classification = re.sub(r"^(Pink|Yellow|Green)\s*:\s*", "", classification, flags=re.I)
                text = (
                    f"Signs: {signs or '(see table)'} -> Classification: {classification} "
                    + f"({IMCI_COLOURS[row['colour']]}) -> Treatment: {treatment}"
                )
                out.append(Piece(text=text, section=section, page_number=pno,
                                 kind="classification_row",
                                 meta={"colour": row["colour"],
                                       # scored on the cell text only, not the template words
                                       "fragment_ratio": round(fragment_ratio(" ".join(texts), vocab), 3)}))

        sec_title = f"{title} > {page_title}"
        tables, table_rects = table_pieces(page, bad_fonts, sec_title) if not rows else ([], [])
        out.extend(tables)
        rest = [l for l in lines if not _inside(l, band_rects + table_rects)]
        if rest:
            text = " ".join(l.text for l in rest if l.text != page_title)
            if text:
                out.append(Piece(text=text, section=sec_title, page_number=pno, kind="text"))
    return out


def extract_document(doc, title: str, pages: Iterable[int]) -> list[Piece]:
    """Generic WHO document: headings tracked across pages, tables as rows, prose."""
    bad_fonts = glyph_encoded_fonts(doc)
    all_lines = {}
    sample = []
    for pno in pages:
        page = doc[pno - 1]
        ls = [l for l in page_lines(page, bad_fonts) if not is_noise(l, page.rect.height)]
        all_lines[pno] = ls
        sample.extend(ls)
    tracker = _SectionTracker(title, body_font_size(sample))
    out: list[Piece] = []
    for pno, lines in all_lines.items():
        page = doc[pno - 1]
        # Section at the top of the page decides the tables' section
        section_now = f"{title} > {tracker.heading}" if tracker.heading else title
        tables, rects = table_pieces(page, bad_fonts, section_now)
        out.extend(tables)
        prose_lines = [l for l in lines if not _inside(l, rects)]
        out.extend(_prose_pieces(prose_lines, pno, tracker))
    return out


# ─── Quality checks ──────────────────────────────────────────────────────────

# Characters left in glyph-shifted text that was not repaired
_GLYPH_MARKS = set("$&*+,<>\\[]^`")


FRAGMENT_THRESHOLD = 0.15


def quality_flags(text: str, fragment: float = 0.0) -> list[str]:
    """
    Reasons a chunk looks garbled (empty list = looks fine).

    - fragmented:       clipped table text where over 15% of words are cut
                        (see fragment_ratio)
    - control_chars:    undecoded glyph ids / control characters
    - replacement_char: U+FFFD from a failed decode
    - glyph_residue:    tokens like '6,&.<281*' (glyph text that was not repaired)
    - missing_spaces:   an alphabetic run over 30 characters (words glued together)
    - low_alpha:        under 55% letters (number grids, charts)
    """
    flags = []
    if fragment > FRAGMENT_THRESHOLD:
        flags.append("fragmented")
    if _looks_glyph_encoded(text):
        flags.append("control_chars")
    if "�" in text:
        flags.append("replacement_char")
    tokens = text.split()
    residue = [t for t in tokens if len(t) >= 6 and not any(c.islower() for c in t)
               and sum(c in _GLYPH_MARKS for c in t) >= 3 and _alpha_ratio(t) < 0.5]
    if len(residue) >= 2:
        flags.append("glyph_residue")
    no_urls = strip_urls(text)
    if re.search(r"[A-Za-z]{31,}", no_urls):
        flags.append("missing_spaces")
    # Glyph ids from a non-linear font map land in Latin Extended / IPA / Greek
    # ('dĂƌŐĞƚ ĚŽƐĞ' = 'Target dose'); real text here has almost none.
    exotic = sum(0x100 <= ord(c) <= 0x3FF for c in text)
    if exotic >= 3 and exotic / max(len(text), 1) > 0.02:
        flags.append("exotic_glyphs")
    if no_urls.strip() and _alpha_ratio(no_urls) < 0.55:
        flags.append("low_alpha")
    return flags


def strip_urls(text: str) -> str:
    """Remove URLs, including ones the PDF wraps over two lines ('.../ 9789...pdf?')."""
    text = re.sub(r"(https?://|www\.)\S+(\s+[\w./-]*(\.pdf|\?sequence=\d+|/)\S*)*", " ", text)
    return re.sub(r"\s+", " ", text).strip()
