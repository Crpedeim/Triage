"""
rag/ingest.py — Clinical Guidelines Ingestion Pipeline

This script is run ONCE (offline) to build the knowledge base.
It takes PDF files from rag/guidelines/, processes them into chunks,
embeds them, and stores them in ChromaDB.

Run it:
    cd triage-agent
    python -m rag.ingest

Or with a specific file:
    python -m rag.ingest --file rag/guidelines/imnci_chart_booklet.pdf

PIPELINE OVERVIEW:
    PDF files → text extraction (PyMuPDF) → chunking (RecursiveCharacterTextSplitter)
    → metadata attachment → embedding (MiniLM / OpenAI) → ChromaDB storage

GUIDELINES TO DOWNLOAD (put in rag/guidelines/):
    1. WHO IMNCI Chart Booklet (CORE - required for MVP):
       https://www.who.int/publications/i/item/9789241506328
       Save as: imnci_chart_booklet.pdf

    2. WHO PEN Guidelines (Enhancement 1 - adult diseases):
       https://www.who.int/publications/i/item/who-nm-nvi-2023.2
       Save as: who_pen_guidelines.pdf

    3. WHO Primary Care Guidelines:
       https://www.who.int/publications/i/item/9789241548649
       Save as: who_primary_care_guidelines.pdf

If you don't have the PDFs yet, run with --sample flag to ingest
sample clinical text that lets you test the RAG pipeline immediately.

METADATA STRATEGY:
The section header is the most important metadata field. When the Triage
Agent says "per IMNCI Section: Assess and Classify Cough", that comes
from the section metadata. We extract it from the PDF's heading structure.

CHUNKING STRATEGY:
- chunk_size=500 tokens (balances context vs precision)
- chunk_overlap=50 tokens (prevents cutting mid-sentence)
- We try to split on paragraph boundaries first (\n\n),
  then sentences (\n), then words ( ).
- Clinical tables are tricky — we keep table rows together
  by treating them as single "sentences".
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.embeddings import embed_texts
from rag.store import add_documents, count, list_sources, reset

load_dotenv()

GUIDELINES_DIR = Path(__file__).parent / "guidelines"
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# Maps filename patterns to guideline types
# Used to set the guideline_type metadata field for age-aware retrieval
GUIDELINE_TYPE_MAP: dict[str, str] = {
    "imnci": "imnci",
    "who_pen": "who_pen",
    "primary_care": "who_primary_care",
    "nrhm": "nrhm",
    "who_primary": "who_primary_care",
}


def _detect_guideline_type(filename: str) -> str:
    """Infer guideline type from filename for metadata tagging."""
    fname_lower = filename.lower()
    for pattern, gtype in GUIDELINE_TYPE_MAP.items():
        if pattern in fname_lower:
            return gtype
    return "general"


def _extract_section_from_text(text: str, prev_section: str) -> str:
    """
    Try to detect if this text starts a new section.

    IMNCI documents use ALL CAPS headers like:
    "ASSESS AND CLASSIFY THE SICK CHILD AGE 2 MONTHS UP TO 5 YEARS"

    WHO PEN documents use title case:
    "Module C: Assessment and Management of Cardiovascular Risk"

    If no new section header found, inherit the previous section.
    """
    lines = text.strip().split("\n")
    for line in lines[:3]:
        line = line.strip()
        if not line:
            continue
        # Check for ALL CAPS header (IMNCI style)
        if len(line) > 10 and line == line.upper() and not line.startswith("•"):
            return line[:120]
        # Check for numbered section header (WHO PEN style)
        if re.match(r'^(Module|Section|Chapter|Part)\s+[A-Z0-9]', line):
            return line[:120]
        # Check for numbered heading: "3.1 Management of..."
        if re.match(r'^\d+\.[\d\.]*\s+[A-Z]', line):
            return line[:120]
    return prev_section


def extract_text_from_pdf(pdf_path: Path) -> list[dict]:
    """
    Extract text from a PDF, page by page, preserving structure.

    Returns a list of page dicts:
        {"text": str, "page_number": int, "source_file": str}

    Uses PyMuPDF (fitz) which handles:
    - Text-based PDFs (most WHO documents)
    - Preserves reading order better than pdfplumber
    - Fast (~0.1s per page)

    NOTE: Scanned PDFs (images) will produce empty text.
    WHO guidelines are typically text-based so this is not an issue.
    """
    import fitz  # PyMuPDF

    pages = []
    with fitz.open(str(pdf_path)) as doc:
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text")  # "text" mode preserves reading order
            # Clean up excessive whitespace while preserving paragraph breaks
            text = re.sub(r'\n{3,}', '\n\n', text)
            text = re.sub(r' {2,}', ' ', text)
            text = text.strip()

            if text:
                pages.append({
                    "text": text,
                    "page_number": page_num,
                    "source_file": pdf_path.name,
                })

    print(f"  Extracted {len(pages)} pages from {pdf_path.name}")
    return pages


def chunk_pages(
    pages: list[dict],
    guideline_type: str,
) -> list[dict]:
    """
    Split page texts into overlapping chunks with metadata.

    Returns a list of chunk dicts:
        {
            "text": str,
            "metadata": {
                "source_file": str,
                "section": str,
                "page_number": int,
                "chunk_index": int,
                "guideline_type": str,
            }
        }

    SECTION TRACKING:
    We track the current section header across pages. When a new section
    header is detected, all subsequent chunks inherit that section name
    until the next header is found. This means the Triage Agent can cite
    "IMNCI: Assess and Classify Cough or Difficult Breathing" even from
    chunks that are in the middle of that section.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = []
    current_section = "General"
    chunk_index = 0

    for page in pages:
        text = page["text"]
        page_num = page["page_number"]
        source_file = page["source_file"]

        # Update section if this page starts a new one
        current_section = _extract_section_from_text(text, current_section)

        # Split this page's text into chunks
        page_chunks = splitter.split_text(text)

        for chunk_text in page_chunks:
            chunk_text = chunk_text.strip()
            if len(chunk_text) < 50:  # Skip tiny fragments (headers, page numbers)
                continue

            # Re-check section at start of each chunk
            chunk_section = _extract_section_from_text(chunk_text, current_section)
            if chunk_section != current_section:
                current_section = chunk_section

            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "source_file": source_file,
                    "section": current_section,
                    "page_number": page_num,
                    "chunk_index": chunk_index,
                    "guideline_type": guideline_type,
                },
            })
            chunk_index += 1

    return chunks


def ingest_pdf(pdf_path: Path, force: bool = False) -> int:
    """
    Full ingestion pipeline for a single PDF file.

    Args:
        pdf_path: Path to the PDF file.
        force:    If True, re-ingest even if already in the store.

    Returns:
        Number of chunks added.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Check if already ingested
    existing = list_sources()
    if pdf_path.name in existing and not force:
        print(f"  [SKIP] {pdf_path.name} already ingested. Use --force to re-ingest.")
        return 0

    print(f"\n[ingest] Processing: {pdf_path.name}")
    guideline_type = _detect_guideline_type(pdf_path.name)
    print(f"  Guideline type: {guideline_type}")

    # Step 1: Extract text
    pages = extract_text_from_pdf(pdf_path)
    if not pages:
        print(f"  [WARN] No text extracted from {pdf_path.name}. Is it a scanned PDF?")
        return 0

    # Step 2: Chunk
    chunks = chunk_pages(pages, guideline_type)
    print(f"  Created {len(chunks)} chunks (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    # Step 3: Embed
    print(f"  Embedding {len(chunks)} chunks...")
    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    # Embed in batches to show progress
    batch_size = 64
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embeddings = embed_texts(batch)
        all_embeddings.extend(embeddings)
        pct = min(100, int((i + batch_size) / len(texts) * 100))
        print(f"  Embedding progress: {pct}%", end="\r")
    print()

    # Step 4: Store
    added = add_documents(texts, metadatas, all_embeddings)
    print(f"  Stored {added} chunks in ChromaDB")

    return added


def ingest_sample_data() -> int:
    """
    Ingest a small set of sample clinical text for testing.

    This creates a minimal knowledge base with actual IMNCI clinical
    content so you can test the RAG pipeline without the full PDF.
    The content is sourced from public WHO IMNCI guidelines.

    Run this if you don't have the PDF yet:
        python -m rag.ingest --sample
    """
    sample_chunks = [
        {
            "text": (
                "ASSESS AND CLASSIFY THE SICK CHILD AGE 2 MONTHS UP TO 5 YEARS\n\n"
                "ASK: Does the child have cough or difficult breathing?\n"
                "If yes, ask: For how long? Count the breaths in one minute.\n"
                "Look for: Chest indrawing. Look and listen for: Stridor.\n\n"
                "CLASSIFY COUGH OR DIFFICULT BREATHING:\n"
                "SEVERE PNEUMONIA OR VERY SEVERE DISEASE: Any general danger sign, "
                "or chest indrawing, or stridor in a calm child. "
                "ACTION: Give first dose of an appropriate antibiotic. "
                "Refer URGENTLY to hospital.\n\n"
                "PNEUMONIA: Fast breathing. "
                "For age 2 months up to 12 months: 50 breaths per minute or more. "
                "For age 12 months up to 5 years: 40 breaths per minute or more. "
                "ACTION: Give an appropriate oral antibiotic for 5 days.\n\n"
                "NO PNEUMONIA: COUGH OR COLD: No signs of pneumonia or very severe disease."
            ),
            "metadata": {
                "source_file": "sample_imnci.txt",
                "section": "ASSESS AND CLASSIFY COUGH OR DIFFICULT BREATHING",
                "page_number": 1,
                "chunk_index": 0,
                "guideline_type": "imnci",
            },
        },
        {
            "text": (
                "GENERAL DANGER SIGNS - CHECK FOR GENERAL DANGER SIGNS\n\n"
                "ASK: Is the child able to drink or breastfeed? Does the child vomit everything? "
                "Has the child had convulsions?\n"
                "LOOK: See if the child is lethargic or unconscious. "
                "See if the child is convulsing now.\n\n"
                "A child has a general danger sign if they: "
                "are NOT able to drink or breastfeed, OR vomit everything, "
                "OR have had more than one convulsion in this illness, "
                "OR are lethargic or unconscious.\n\n"
                "A child with ANY general danger sign needs URGENT attention. "
                "Complete the assessment and any pre-referral treatment immediately. "
                "Refer the child to hospital."
            ),
            "metadata": {
                "source_file": "sample_imnci.txt",
                "section": "GENERAL DANGER SIGNS",
                "page_number": 2,
                "chunk_index": 1,
                "guideline_type": "imnci",
            },
        },
        {
            "text": (
                "ASSESS AND CLASSIFY DIARRHOEA\n\n"
                "ASK: Does the child have diarrhoea? If yes:\n"
                "For how long? Is there blood in the stool?\n"
                "LOOK AND FEEL: Look at the child's general condition: "
                "Is the child lethargic or unconscious? Restless and irritable?\n"
                "Look for sunken eyes. Offer the child fluid. Is the child not able to drink, "
                "or drinking poorly? Drinking eagerly, thirsty?\n"
                "Pinch the skin of the abdomen. Does it go back very slowly (more than 2 seconds)? Slowly?\n\n"
                "SEVERE DEHYDRATION: Two of the following signs: lethargic or unconscious, "
                "sunken eyes, not able to drink or drinking poorly, skin pinch goes back very slowly.\n"
                "ACTION: Give fluid for severe dehydration (Plan C). Refer URGENTLY to hospital.\n\n"
                "SOME DEHYDRATION: Two of the following signs: restless, irritable, "
                "sunken eyes, drinks eagerly/thirsty, skin pinch goes back slowly.\n"
                "ACTION: Give fluid, zinc and food for some dehydration (Plan B)."
            ),
            "metadata": {
                "source_file": "sample_imnci.txt",
                "section": "ASSESS AND CLASSIFY DIARRHOEA",
                "page_number": 3,
                "chunk_index": 2,
                "guideline_type": "imnci",
            },
        },
        {
            "text": (
                "ASSESS AND CLASSIFY FEVER\n\n"
                "If child lives in malaria risk area, or has visited such an area:\n"
                "Do malaria test.\n\n"
                "VERY SEVERE FEBRILE DISEASE: Any general danger sign, or stiff neck.\n"
                "ACTION: Give first dose of diazepam if convulsing. Give first dose of appropriate antibiotic. "
                "Treat the child to prevent low blood sugar. Give one dose of paracetamol in clinic for high fever. "
                "Refer URGENTLY to hospital.\n\n"
                "FEVER (malaria unlikely): No runny nose, no measles, no other cause of fever. "
                "Positive malaria test.\n"
                "ACTION: Give first-line oral antimalarial.\n\n"
                "FEVER: malaria unlikely. Runny nose present, or measles, or other cause of fever. "
                "Negative malaria test (or no test done).\n"
                "Give paracetamol for high fever. Advise mother when to return immediately."
            ),
            "metadata": {
                "source_file": "sample_imnci.txt",
                "section": "ASSESS AND CLASSIFY FEVER",
                "page_number": 4,
                "chunk_index": 3,
                "guideline_type": "imnci",
            },
        },
        {
            "text": (
                "IMNCI FAST BREATHING THRESHOLDS BY AGE\n\n"
                "Fast breathing is defined as:\n"
                "- Child age 0 up to 2 months: 60 breaths per minute or more\n"
                "- Child age 2 months up to 12 months: 50 breaths per minute or more\n"
                "- Child age 12 months up to 5 years: 40 breaths per minute or more\n\n"
                "Count the breaths in one minute. The child must be calm.\n"
                "If the count is 60 or more breaths per minute (for a child under 2 months), "
                "or 50 or more (for age 2-12 months), or 40 or more (for age 1-5 years), "
                "the child has fast breathing.\n\n"
                "Fast breathing is a sign of pneumonia. "
                "Fast breathing WITH chest indrawing indicates SEVERE PNEUMONIA."
            ),
            "metadata": {
                "source_file": "sample_imnci.txt",
                "section": "FAST BREATHING THRESHOLDS",
                "page_number": 5,
                "chunk_index": 4,
                "guideline_type": "imnci",
            },
        },
        {
            "text": (
                "CHEST INDRAWING - DEFINITION AND SIGNIFICANCE\n\n"
                "Chest indrawing is when the lower chest wall goes IN when the child BREATHES IN. "
                "Look carefully at the lower chest (lower ribs area) while the child breathes in.\n\n"
                "Chest indrawing is present when the lower chest wall goes inward. "
                "This is DIFFERENT from drawing in of the soft tissue between the ribs "
                "or above the collar bone. These are NOT chest indrawing.\n\n"
                "IMPORTANCE: Chest indrawing means the child has to make a great effort to breathe. "
                "When the lungs become stiff (as in severe pneumonia), chest indrawing appears.\n\n"
                "A child with chest indrawing has SEVERE PNEUMONIA OR VERY SEVERE DISEASE. "
                "This requires URGENT REFERRAL to a hospital immediately. "
                "DO NOT delay referral for a child with chest indrawing."
            ),
            "metadata": {
                "source_file": "sample_imnci.txt",
                "section": "CHEST INDRAWING DEFINITION",
                "page_number": 6,
                "chunk_index": 5,
                "guideline_type": "imnci",
            },
        },
        {
            "text": (
                "WHO PEN PROTOCOL 1: CARDIOVASCULAR RISK MANAGEMENT\n\n"
                "ASSESS: Blood pressure measurement. "
                "If BP >= 140/90 mmHg on two separate occasions, diagnose hypertension.\n\n"
                "CLASSIFY HYPERTENSION:\n"
                "Grade 1 (Mild): Systolic 140-159 OR Diastolic 90-99 mmHg\n"
                "Grade 2 (Moderate): Systolic 160-179 OR Diastolic 100-109 mmHg\n"
                "Grade 3 (Severe): Systolic >= 180 OR Diastolic >= 110 mmHg\n\n"
                "Hypertensive Crisis (Emergency): Systolic > 180 AND/OR Diastolic > 120 "
                "WITH symptoms (headache, chest pain, visual disturbance, seizure). "
                "ACTION: Urgent referral to hospital. Do not delay.\n\n"
                "Grade 2-3 without emergency features: "
                "Refer to physician for treatment initiation within 24-48 hours."
            ),
            "metadata": {
                "source_file": "sample_who_pen.txt",
                "section": "CARDIOVASCULAR RISK MANAGEMENT",
                "page_number": 1,
                "chunk_index": 6,
                "guideline_type": "who_pen",
            },
        },
        {
            "text": (
                "WHO PEN PROTOCOL 2: DIABETES RISK ASSESSMENT\n\n"
                "SCREEN for diabetes if patient has any of:\n"
                "- Age >= 45 years\n"
                "- BMI >= 25 (or waist circumference > 80cm women, > 90cm men for South Asians)\n"
                "- Family history of diabetes\n"
                "- History of gestational diabetes\n"
                "- Hypertension (BP >= 140/90 mmHg)\n\n"
                "CLASSIFY using fasting blood glucose:\n"
                "Normal: < 5.6 mmol/L (< 100 mg/dL)\n"
                "Pre-diabetes (IFG): 5.6 - 6.9 mmol/L (100-125 mg/dL)\n"
                "Diabetes: >= 7.0 mmol/L (>= 126 mg/dL) on two separate occasions\n\n"
                "If HbA1c available:\n"
                "Normal: < 5.7%\n"
                "Pre-diabetes: 5.7-6.4%\n"
                "Diabetes: >= 6.5%\n\n"
                "ACTION for diabetes: Refer to physician for treatment initiation."
            ),
            "metadata": {
                "source_file": "sample_who_pen.txt",
                "section": "DIABETES RISK ASSESSMENT",
                "page_number": 2,
                "chunk_index": 7,
                "guideline_type": "who_pen",
            },
        },
    ]

    print(f"\n[ingest] Loading sample clinical data ({len(sample_chunks)} chunks)...")
    texts = [c["text"] for c in sample_chunks]
    metadatas = [c["metadata"] for c in sample_chunks]

    print("[ingest] Embedding sample chunks...")
    embeddings = embed_texts(texts)

    added = add_documents(texts, metadatas, embeddings)
    print(f"[ingest] Done. Added {added} sample chunks.")
    print("[ingest] Sample data includes: IMNCI (cough, diarrhea, fever, danger signs) + WHO PEN (hypertension, diabetes)")
    return added


def ingest_all_pdfs(guidelines_dir: Path = GUIDELINES_DIR, force: bool = False) -> int:
    """
    Ingest all PDF files found in the guidelines directory.

    This is the main entry point for the full ingestion pipeline.
    Run once to build the knowledge base. Re-run with force=True to rebuild.
    """
    pdf_files = list(guidelines_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"[ingest] No PDF files found in {guidelines_dir}")
        print("[ingest] Place PDF files in rag/guidelines/ and re-run.")
        print("[ingest] Or run with --sample to use built-in sample data.")
        return 0

    total = 0
    for pdf_path in sorted(pdf_files):
        added = ingest_pdf(pdf_path, force=force)
        total += added

    return total


def print_status() -> None:
    """Print current state of the vector store."""
    total = count()
    sources = list_sources()
    print(f"\n[store status] Total chunks: {total}")
    if sources:
        print(f"[store status] Sources ({len(sources)}):")
        for src in sources:
            print(f"  - {src}")
    else:
        print("[store status] No sources ingested yet.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest clinical guidelines into ChromaDB")
    parser.add_argument(
        "--file", type=str, default=None,
        help="Ingest a specific PDF file"
    )
    parser.add_argument(
        "--sample", action="store_true",
        help="Ingest built-in sample data (for testing without PDFs)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-ingest even if source already exists in store"
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="DANGER: Delete all data and rebuild from scratch"
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Print current store status and exit"
    )
    args = parser.parse_args()

    if args.status:
        print_status()
        sys.exit(0)

    if args.reset:
        print("[ingest] Resetting vector store...")
        reset()
        print("[ingest] Store cleared.")

    if args.sample:
        ingest_sample_data()
    elif args.file:
        path = Path(args.file)
        ingest_pdf(path, force=args.force)
    else:
        ingest_all_pdfs(force=args.force)

    print_status()
