"""
rag/build_kb.py — build the guideline knowledge base from the WHO PDFs.

Runs at Docker BUILD time (see Dockerfile), never at app startup:

    python -m rag.fetch_guidelines        # download + sha256-check the PDFs
    python -m rag.build_kb                # extract, check, chunk, embed, store

Other modes:
    python -m rag.build_kb --dry-run --inspect 20   # no embedding; print 20
                                                    # random chunks per document
    python -m rag.build_kb --stats-only             # token stats for chunk tuning

Pipeline per document (rag/guidelines/sources.json):
  1. Page roles from the manifest (front matter, recording forms, charts) are
     skipped.
  2. Table-aware extraction (rag/pdf_extract.py) gives pieces of text, each
     with its section heading.
  3. Pieces are chunked; every chunk starts with its section heading.
  4. Quality check: each chunk gets garble flags. A page where more than
     MAX_GARBLED_FRACTION of the chunks are flagged is rejected as a whole;
     on accepted pages, flagged chunks are dropped individually.
  5. Curated fallback: a hand-written excerpt from rag/ingest.py is kept only
     if one of the pages that covers its topic was rejected.
  6. Embed (all-MiniLM-L6-v2) and store in ChromaDB, replacing what was there.

A JSON report (pages used / replaced / skipped, chunk counts, token stats) is
written next to the vector store: <CHROMA_PERSIST_DIR>/kb_report.json.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.fetch_guidelines import GUIDELINES_DIR, is_valid, load_manifest
from rag.pdf_extract import Piece, extract_document, extract_imci_chart_booklet, quality_flags, strip_urls

load_dotenv()

# Chunk size in characters, heading excluded. Tuned so that a chunk plus its
# heading fits the embedding model's 256-token window (see PHASES.md, R2 Phase 1).
CHUNK_SIZE = int(os.getenv("KB_CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("KB_CHUNK_OVERLAP", "120"))
MAX_GARBLED_FRACTION = 0.3
MIN_CHUNK_CHARS = 60


def _ranges_to_pages(ranges) -> set[int]:
    return {p for a, b in ranges for p in range(a, b + 1)}


def page_roles(doc_entry: dict) -> dict[int, str]:
    roles = {}
    for role, ranges in doc_entry.get("page_roles", {}).items():
        for p in _ranges_to_pages(ranges):
            roles[p] = role
    return roles


def extract_pieces(doc_entry: dict, pdf, pages: list[int]) -> list[Piece]:
    title = doc_entry["short_title"]
    if doc_entry["extractor"] == "imci_chart_booklet":
        return extract_imci_chart_booklet(pdf, title, pages)
    return extract_document(pdf, title, pages)


def _merge_prose(pieces: list[Piece]) -> list[Piece]:
    """Join consecutive prose pieces of the same section (a section often spans pages)."""
    out: list[Piece] = []
    for p in pieces:
        prev = out[-1] if out else None
        if (prev and p.kind == "text" and prev.kind == "text" and p.section == prev.section
                and len(prev.text) + len(p.text) < CHUNK_SIZE * 3):
            prev.text = f"{prev.text} {p.text}"
            prev.meta.setdefault("pages", [prev.page_number]).append(p.page_number)
        else:
            out.append(Piece(text=p.text, section=p.section, page_number=p.page_number,
                             kind=p.kind, meta=dict(p.meta)))
    return out


_ENCODING_FLAGS = {"control_chars", "replacement_char", "glyph_residue", "exotic_glyphs", "missing_spaces"}


def clean_section(section: str) -> str:
    """
    Drop heading parts that are themselves encoding garbage
    ('WHO PEN (2020) > PNEõ//,/kkmj-J&0..' -> 'WHO PEN (2020)'). The document
    title is always kept, and short numeric headings ('2.3') are not garbage.
    """
    title, *rest = section.split(" > ")
    rest = [p for p in rest if p and not (set(quality_flags(p)) & _ENCODING_FLAGS)
            and sum(c.isalpha() for c in p) >= 3]
    return " > ".join([title, *rest])


def chunk_pieces(pieces: list[Piece], doc_entry: dict) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "; ", " ", ""], keep_separator="end",
        length_function=len,
    )
    chunks = []
    for piece in _merge_prose(pieces):
        bodies = [piece.text] if len(piece.text) <= CHUNK_SIZE else splitter.split_text(piece.text)
        section = clean_section(piece.section)
        for body in bodies:
            body = body.strip().lstrip(".;, ")
            # too short, or nothing but a reference URL
            if len(body) < MIN_CHUNK_CHARS or len(strip_urls(body)) < MIN_CHUNK_CHARS:
                continue
            pages = piece.meta.get("pages", [piece.page_number])
            chunks.append({
                "text": f"{section}\n{body}",
                "body": body,
                "metadata": {
                    "source_file": doc_entry["file"],
                    "document": doc_entry["short_title"],
                    "section": section,
                    "page_number": piece.page_number,
                    "pages": f"{min(pages)}-{max(pages)}" if len(set(pages)) > 1 else str(piece.page_number),
                    "kind": piece.kind,
                    "guideline_type": doc_entry["guideline_type"],
                    "colour": piece.meta.get("colour", ""),
                },
                "flags": quality_flags(body, piece.meta.get("fragment_ratio", 0.0)),
            })
    return chunks


def build(dry_run: bool = False, inspect: int = 0, seed: int = 7) -> dict:
    import pymupdf

    t0 = time.perf_counter()
    manifest = load_manifest()
    report: dict = {"chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP,
                    "max_garbled_fraction": MAX_GARBLED_FRACTION, "documents": []}
    kept_chunks: list[dict] = []
    page_status: dict[str, dict[int, str]] = {}

    for doc_entry in manifest["documents"]:
        path = GUIDELINES_DIR / doc_entry["file"]
        if not is_valid(path, doc_entry["sha256"]):
            raise SystemExit(f"[build_kb] {doc_entry['file']} missing or hash mismatch; "
                             "run: python -m rag.fetch_guidelines")
        pdf = pymupdf.open(str(path))
        roles = page_roles(doc_entry)
        content_pages = [p for p in range(1, pdf.page_count + 1) if p not in roles]
        pieces = extract_pieces(doc_entry, pdf, content_pages)
        chunks = chunk_pieces(pieces, doc_entry)

        # Chunks covering each page (prose merged across pages covers all of them)
        covering: dict[int, list[dict]] = defaultdict(list)
        for c in chunks:
            a, _, b = c["metadata"]["pages"].partition("-")
            for p in range(int(a), int(b or a) + 1):
                covering[p].append(c)

        status: dict[int, str] = {}
        for p in range(1, pdf.page_count + 1):
            cs = covering.get(p, [])
            if p in roles:
                status[p] = f"skipped:{roles[p]}"
            elif not cs:
                status[p] = "skipped:no_text"  # nothing of MIN_CHUNK_CHARS or more
            # A fragmented table row is dropped on its own; it does not make the
            # page's prose bad. Every other garble type counts against the page.
            elif sum(bool(set(c["flags"]) - {"fragmented"}) for c in cs) / len(cs) > MAX_GARBLED_FRACTION:
                status[p] = "bad"
            else:
                status[p] = "used"
        page_status[doc_entry["id"]] = status

        accepted = [c for c in chunks if status.get(c["metadata"]["page_number"]) == "used"]
        dropped_flagged = [c for c in accepted if c["flags"]]
        accepted = [c for c in accepted if not c["flags"]]
        kept_chunks.extend(accepted)

        flag_counts = Counter(f for c in chunks for f in c["flags"])
        kinds = Counter(c["metadata"]["kind"] for c in accepted)
        report["documents"].append({
            "id": doc_entry["id"], "file": doc_entry["file"], "pages_total": pdf.page_count,
            "page_status": {str(k): v for k, v in status.items()},
            "chunks_extracted": len(chunks), "chunks_kept": len(accepted),
            "chunks_dropped_flagged_on_used_pages": len(dropped_flagged),
            "chunk_kinds_kept": dict(kinds), "flag_counts_all_chunks": dict(flag_counts),
        })

        if inspect:
            rng = random.Random(f"{seed}:{doc_entry['id']}")
            sample = rng.sample(chunks, min(inspect, len(chunks)))
            print(f"\n{'=' * 100}\n[inspect] {doc_entry['short_title']}: {len(sample)} random chunks "
                  f"of {len(chunks)} (seed {seed})\n{'=' * 100}")
            for i, c in enumerate(sample, 1):
                m = c["metadata"]
                print(f"\n--- #{i} p{m['pages']} {m['kind']} flags={c['flags'] or 'none'} "
                      f"page_status={status.get(m['page_number'])}\n{c['text'][:700]}")

    # ─── Curated fallback ────────────────────────────────────────────────────
    from rag.ingest import curated_chunks

    curated_kept, replaced_pages = [], defaultdict(set)
    for cc in curated_chunks():
        topic = cc["metadata"]["section"]
        covering = manifest["curated_fallback"].get(topic, {})
        bad = {(doc_id, p) for doc_id, ps in covering.items() for p in ps
               if page_status.get(doc_id, {}).get(p) != "used"}
        if not covering or bad:
            curated_kept.append(cc)
            for doc_id, p in bad:
                replaced_pages[doc_id].add(p)
    for c in curated_kept:
        kept_chunks.append({"text": c["text"], "body": c["text"],
                            "metadata": {**c["metadata"], "kind": "curated", "document": "curated excerpt",
                                         "pages": str(c["metadata"]["page_number"]), "colour": ""},
                            "flags": []})

    # ─── Page tallies ────────────────────────────────────────────────────────
    for d in report["documents"]:
        st = page_status[d["id"]]
        replaced = {p for p, s in st.items() if s == "bad" and p in replaced_pages[d["id"]]}
        d["pages_used"] = sum(s == "used" for s in st.values())
        d["pages_replaced_by_curated"] = len(replaced)
        d["pages_skipped"] = sum(s != "used" for s in st.values()) - len(replaced)
        d["pages_replaced_list"] = sorted(replaced)
        d["pages_bad_not_replaced"] = sorted(p for p, s in st.items() if s == "bad" and p not in replaced)
        d["skip_reasons"] = dict(Counter(s.split(":", 1)[1] if ":" in s else "garbled"
                                         for p, s in st.items() if s != "used" and p not in replaced))
    report["curated_kept"] = [c["metadata"]["section"] for c in curated_kept]
    report["chunks_total"] = len(kept_chunks)
    report["chunks_by_guideline_type"] = dict(Counter(c["metadata"]["guideline_type"] for c in kept_chunks))
    report["chunk_chars"] = _stats([len(c["text"]) for c in kept_chunks])
    report["token_stats"] = _token_stats([c["text"] for c in kept_chunks])
    report["extract_seconds"] = round(time.perf_counter() - t0, 2)

    if dry_run:
        return report

    # ─── Embed + store ───────────────────────────────────────────────────────
    from rag.embeddings import embed_texts
    from rag.store import CHROMA_PERSIST_DIR, add_documents, count, reset

    t1 = time.perf_counter()
    reset()
    texts = [c["text"] for c in kept_chunks]
    metas = [{**c["metadata"], "chunk_index": i} for i, c in enumerate(kept_chunks)]
    embeddings = []
    for i in range(0, len(texts), 64):
        embeddings.extend(embed_texts(texts[i:i + 64]))
    add_documents(texts, metas, embeddings)
    report["embed_store_seconds"] = round(time.perf_counter() - t1, 2)
    report["store_count"] = count()
    out = Path(CHROMA_PERSIST_DIR) / "kb_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[build_kb] report written to {out}")
    return report


def _stats(values: list[int]) -> dict:
    if not values:
        return {}
    v = sorted(values)
    pick = lambda q: v[min(len(v) - 1, int(round(q * (len(v) - 1))))]
    return {"n": len(v), "min": v[0], "median": pick(0.5), "p95": pick(0.95), "max": v[-1]}


def _token_stats(texts: list[str]) -> dict:
    """Token counts with the embedding model's own tokenizer (local provider only)."""
    if os.getenv("EMBEDDING_PROVIDER", "local") != "local":
        return {}
    from rag.embeddings import _get_local_model

    model = _get_local_model()
    limit = model.max_seq_length
    counts = [len(model.tokenizer(t, add_special_tokens=True)["input_ids"]) for t in texts]
    s = _stats(counts)
    s["max_seq_length"] = limit
    s["over_limit"] = sum(c > limit for c in counts)
    return s


def _print_summary(report: dict) -> None:
    print(f"\n[build_kb] chunk_size={report['chunk_size']} overlap={report['chunk_overlap']}")
    for d in report["documents"]:
        print(f"  {d['file']}: {d['pages_total']} pages -> used {d['pages_used']}, "
              f"replaced {d['pages_replaced_by_curated']}, skipped {d['pages_skipped']} {d['skip_reasons']}; "
              f"chunks kept {d['chunks_kept']} of {d['chunks_extracted']} "
              f"(dropped flagged on used pages: {d['chunks_dropped_flagged_on_used_pages']}); "
              f"bad pages {d['pages_bad_not_replaced'] + d['pages_replaced_list']}")
    print(f"  curated excerpts kept: {report['curated_kept']}")
    print(f"  total chunks: {report['chunks_total']} {report['chunks_by_guideline_type']}")
    print(f"  chunk chars: {report['chunk_chars']}")
    print(f"  tokens: {report['token_stats']}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Build the WHO guideline knowledge base")
    parser.add_argument("--dry-run", action="store_true", help="extract and check only; no embedding")
    parser.add_argument("--inspect", type=int, default=0, help="print N random chunks per document")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--report", type=str, default="", help="also write the report JSON here")
    args = parser.parse_args()
    rep = build(dry_run=args.dry_run, inspect=args.inspect, seed=args.seed)
    _print_summary(rep)
    if args.report:
        Path(args.report).write_text(json.dumps(rep, indent=2), encoding="utf-8")
