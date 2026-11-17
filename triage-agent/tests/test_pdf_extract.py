"""
Offline tests for the WHO guideline extraction pipeline
(rag/pdf_extract.py, rag/build_kb.py, rag/guidelines/sources.json).

No PDFs from WHO are needed: the classification-row extractor is tested on a
synthetic chart page drawn with PyMuPDF, and the glyph repair on strings taken
from the real documents.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag.pdf_extract import (
    decode_glyphs,
    extract_imci_chart_booklet,
    fragment_ratio,
    is_line_split,
    quality_flags,
    repair_glyph_text,
    strip_urls,
)

pymupdf = pytest.importorskip("pymupdf")


# ─── Glyph-id repair (strings from the real PDFs) ────────────────────────────

class TestGlyphRepair:
    def test_glyph3_space_span_is_decoded(self):
        # IMCI chart booklet p4 heading, as PyMuPDF returns it
        raw = "6,&.\x03<281*\x03,1)$17\x03$*(\x0383\x0372\x03\x15\x030217+6"
        assert repair_glyph_text(raw) == "SICK YOUNG INFANT AGE UP TO 2 MONTHS"

    def test_pen_span_with_ligature_and_symbol(self):
        raw = "\x03\x03V\\VWROLF\x03EORRG\x03SUHVVXUH\x03LV\x03Ȳ\x03\x14\x17\x13\x03PP+J"
        assert repair_glyph_text(raw).strip() == "systolic blood pressure is ≥ 140 mmHg"
        assert decode_glyphs("GLɝFXOW") == "difficult"

    def test_font_rule_decodes_single_shifted_word(self):
        assert repair_glyph_text(",1)(&7,21", font_is_glyph_encoded=True) == "INFECTION"
        assert repair_glyph_text("',$55+2($", font_is_glyph_encoded=True) == "DIARRHOEA"

    @pytest.mark.parametrize("text", ["2020", "2019–2020.", "(PEN)", "*", "< 5%",
                                      "ISBN 978-92-4-000922-6", "https://creativecommons.org"])
    def test_font_rule_leaves_correct_text_alone(self, text):
        # WHO PEN uses the same fonts for correctly encoded digits, ISBNs and URLs
        assert repair_glyph_text(text, font_is_glyph_encoded=True) == text

    def test_plain_text_untouched(self):
        assert repair_glyph_text("Chest indrawing or fast breathing.") == "Chest indrawing or fast breathing."


# ─── Quality flags ───────────────────────────────────────────────────────────

class TestQualityFlags:
    def test_clean_text_has_no_flags(self):
        assert quality_flags("Signs: Chest indrawing or Fast breathing. -> Classification: PNEUMONIA") == []

    def test_unrepaired_glyph_text_is_flagged(self):
        assert "glyph_residue" in quality_flags("6,&.<281*,1)$17$*(83720217+6 $66(66$1'&/$66,)<7+(")

    def test_control_chars_flagged(self):
        assert "control_chars" in quality_flags("abc\x03def ghi jkl mno pqr")

    def test_exotic_glyphs_flagged(self):
        assert "exotic_glyphs" in quality_flags("LOPINAVIR dĂƌŐĞƚ ĚŽƐĞ ϮϯϬͲϯϱϬŵŐͬ twice daily")

    def test_missing_spaces_flagged(self):
        assert "missing_spaces" in quality_flags("Givefirstdoseofanappropriateantibiotic now")

    def test_number_grid_is_low_alpha_but_urls_are_ignored(self):
        assert "low_alpha" in quality_flags("4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 mmol/l 20")
        assert quality_flags("Counselling for healthy lifestyles https://apps.who.int/iris/bitstream/"
                             "handle/10665/112833/ 9789241506939_eng.pdf?sequence=1") == []

    def test_fragmented_flag_uses_page_vocabulary(self):
        vocab = {"assess", "for", "signs", "and", "symptoms", "itchy", "nodule"}
        assert fragment_ratio("Assess for signs and symptoms", vocab) == 0.0
        clipped = "SESS minal tchy co ore MPTOMS"
        assert fragment_ratio(clipped, vocab) > 0.15
        assert "fragmented" in quality_flags(clipped, fragment=fragment_ratio(clipped, vocab))

    def test_strip_urls_handles_wrapped_links(self):
        assert strip_urls("see https://apps.who.int/iris/bitstream/ handle/1/2/x.pdf?sequence=1 now") == "see now"


# ─── Table shape ─────────────────────────────────────────────────────────────

class TestTableShape:
    def test_dose_table_is_not_line_split(self):
        rows = [["2 up to 4 months", "1 ml", "0.5 ml"], ["4 up to 12 months", "2 ml", "1.1 ml"],
                ["12 months up to 3 years", "3 ml", "1.9 ml"], ["3 up to 5 years", "5 ml", "2.8 ml"]]
        assert not is_line_split(rows)

    def test_one_row_per_line_table_is_line_split(self):
        # skin-problem table: CLASSIFY AS filled once per condition, signs wrap over lines
        rows = [["Vesicles over body.", "CHICKEN POX", "Treat itching"],
                ["Vesicles appear", "", "Refer if pneumonia"],
                ["progressively over days", "", ""],
                ["Vesicles in one area", "HERPES ZOSTER", "Keep lesions clean"],
                ["with intense pain", "", "Give pain relief"],
                ["shooting pain.", "", ""]]
        assert is_line_split(rows)


# ─── Classification rows on a synthetic chart page ───────────────────────────

PINK, YELLOW, GREEN = (1.0, 0.67, 0.67), (1.0, 1.0, 0.67), (0.67, 1.0, 0.67)


@pytest.fixture
def chart_pdf():
    """A landscape page laid out like an IMCI chart: question + 3 coloured rows."""
    doc = pymupdf.open()
    page = doc.new_page(width=842, height=595)
    page.insert_text((30, 30), "THEN ASK ABOUT MAIN SYMPTOMS:", fontsize=14, fontname="hebo")
    page.insert_text((30, 52), "Does the child have cough or difficult breathing?", fontsize=10, fontname="hebo")
    rows = [
        (PINK, 60, 120, "Any general danger sign or Stridor in calm child.",
         "Pink: SEVERE PNEUMONIA OR VERY SEVERE DISEASE", "Give first dose of an appropriate antibiotic. Refer URGENTLY."),
        (YELLOW, 120, 200, "Chest indrawing or Fast breathing.",
         "Yellow: PNEUMONIA", "Give oral Amoxicillin for 5 days."),
        (GREEN, 200, 260, "No signs of pneumonia or very severe disease.",
         "Green: COUGH OR COLD", "Soothe the throat and relieve the cough."),
    ]
    for colour, y0, y1, signs, cls, treat in rows:
        for (x0, x1), text in zip(((405, 525), (525, 605), (605, 804)), (signs, cls, treat)):
            r = pymupdf.Rect(x0, y0, x1, y1)
            page.draw_rect(r, color=(0, 0, 0), fill=colour)
            page.insert_textbox(r + (3, 3, -3, -3), text, fontsize=8)
    page.insert_text((30, 150), "Count the breaths in one minute. Look for chest indrawing.", fontsize=8)
    yield doc
    doc.close()


def test_classification_rows_from_coloured_cells(chart_pdf):
    pieces = extract_imci_chart_booklet(chart_pdf, "IMCI chart", [1])
    rows = [p for p in pieces if p.kind == "classification_row"]
    assert [p.meta["colour"] for p in rows] == ["pink", "yellow", "green"]
    pneumonia = rows[1].text
    assert "Signs: Chest indrawing or Fast breathing." in pneumonia
    assert "Classification: PNEUMONIA (Yellow:" in pneumonia
    assert "Treatment: Give oral Amoxicillin for 5 days." in pneumonia
    assert "SEVERE PNEUMONIA OR VERY SEVERE DISEASE" in rows[0].text
    # section heading: document > page title > symptom question
    assert rows[0].section == ("IMCI chart > THEN ASK ABOUT MAIN SYMPTOMS > "
                               "Does the child have cough or difficult breathing?")
    for p in rows:
        assert quality_flags(p.text, p.meta["fragment_ratio"]) == []
    # assessment text outside the coloured cells is kept as prose, once
    prose = [p for p in pieces if p.kind == "text"]
    assert any("Count the breaths in one minute" in p.text for p in prose)
    assert not any("Amoxicillin" in p.text for p in prose)


# ─── build_kb helpers ────────────────────────────────────────────────────────

def test_clean_section_drops_glyph_soup_keeps_title():
    from rag.build_kb import clean_section

    # WHO PEN p30 heading as extracted (a second, \x01-spaced glyph encoding)
    raw = "PNE\x01õ\x01//,/\x01kkmj-J\x01&0..ke-bj'\x01_lb\x01:/6\x01kkmj-J"
    assert clean_section(f"WHO PEN (2020) > {raw}") == "WHO PEN (2020)"
    assert clean_section("WHO PEN (2020) > HYPERTENSION") == "WHO PEN (2020) > HYPERTENSION"


def test_manifest_is_complete_and_age_filter_types_are_valid():
    manifest = json.loads((Path(__file__).parent.parent / "rag" / "guidelines" / "sources.json")
                          .read_text(encoding="utf-8"))
    docs = manifest["documents"]
    assert {d["guideline_type"] for d in docs} == {"imnci", "who_pen"}  # the two retrieval filters
    for d in docs:
        assert d["pdf_url"].startswith("https://iris.who.int/")
        assert len(d["sha256"]) == 64 and d["licence"]
        assert d["iris_handle"].startswith("https://iris.who.int/handle/10665/")
    ids = {d["id"] for d in docs}
    for topic, covering in manifest["curated_fallback"].items():
        if topic.startswith("_"):
            continue
        assert set(covering) <= ids, topic


def test_curated_topics_match_curated_excerpts():
    from rag.ingest import curated_chunks

    manifest = json.loads((Path(__file__).parent.parent / "rag" / "guidelines" / "sources.json")
                          .read_text(encoding="utf-8"))
    topics = {k for k in manifest["curated_fallback"] if not k.startswith("_")}
    assert topics == {c["metadata"]["section"] for c in curated_chunks()}
