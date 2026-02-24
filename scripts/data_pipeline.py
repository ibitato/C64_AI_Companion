#!/usr/bin/env python3
"""
End-to-end data pipeline for C64 document preparation.

Stages:
  - manifest
  - extract
  - normalize
  - dedup
  - build_dapt
  - build_sft
  - validate
  - all
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import random
import re
import shutil
import subprocess
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from difflib import SequenceMatcher

import pandas as pd
import pypdf
from bs4 import BeautifulSoup
from transformers import AutoTokenizer


DEFAULT_SOURCE_DIR = Path("c64_docs")
DEFAULT_MODEL_PATH = Path("models/Ministral-3-8B-Thinking")

INTERIM_DIR = Path("data/interim")
PROCESSED_DIR = Path("data/processed")

MANIFEST_PATH = INTERIM_DIR / "manifest" / "manifest.parquet"
EXTRACTED_PATH = INTERIM_DIR / "extracted" / "pages.parquet"
NORMALIZED_PATH = INTERIM_DIR / "normalized" / "pages_normalized.parquet"
DEDUP_PATH = INTERIM_DIR / "dedup" / "pages_dedup.parquet"
DUPLICATES_PATH = INTERIM_DIR / "dedup" / "duplicates.parquet"

VALIDATION_REPORT_PATH = PROCESSED_DIR / "validation_report.json"

C64_SYSTEM_PROMPT = (
    "You are a specialized Commodore 64 technical assistant.\n\n"
    "Scope:\n"
    "- Only answer Commodore 64 and directly related topics: C64 hardware specs, "
    "memory map, VIC-II, SID, CIA, KERNAL, BASIC V2, 6502/6510 machine language, "
    "programming, debugging, and emulation.\n\n"
    "Behavior:\n"
    "- Be concise, precise, and polite.\n"
    "- Prefer short, practical answers.\n"
    "- If a request is outside scope, say it briefly and ask for a C64-focused question.\n"
    "- If information is uncertain, state uncertainty and avoid guessing.\n"
    "- Respond in the same language as the user."
)


def log(msg: str) -> None:
    print(f"[data_pipeline] {msg}")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def list_source_files(source_dir: Path) -> list[Path]:
    files: list[Path] = []
    for ext in ("*.pdf", "*.htm", "*.html"):
        files.extend(sorted(source_dir.rglob(ext)))
    return files


def text_quality_score(text: str) -> float:
    if not text:
        return 0.0
    t = text.strip()
    if not t:
        return 0.0

    length_component = min(len(t), 5000) / 5000
    printable = sum(ch.isprintable() for ch in t) / len(t)
    alnum = sum(ch.isalnum() or ch.isspace() for ch in t) / len(t)
    weird = sum(ch in "\ufffd\u0000" for ch in t) / len(t)
    words = t.split()
    long_words_ratio = 0.0
    if words:
        long_words_ratio = sum(len(w) > 30 for w in words) / len(words)

    score = (
        0.40 * length_component
        + 0.30 * printable
        + 0.30 * alnum
        - 0.35 * weird
        - 0.20 * long_words_ratio
    )
    return round(max(0.0, min(1.0, score)), 4)


def has_command(name: str) -> bool:
    return shutil.which(name) is not None


def safe_extract_page_text(page: Any) -> str:
    # pypdf can emit noisy warnings to stdout/stderr depending on source PDFs.
    sink_out = io.StringIO()
    sink_err = io.StringIO()
    try:
        with contextlib.redirect_stdout(sink_out), contextlib.redirect_stderr(sink_err):
            text = page.extract_text()
        return text or ""
    except Exception:
        return ""


def extract_pdf_digital(pdf_path: Path) -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []
    with pdf_path.open("rb") as f:
        reader = pypdf.PdfReader(f)
        for i, page in enumerate(reader.pages, start=1):
            text = safe_extract_page_text(page)
            pages.append(
                {
                    "page_number": i,
                    "text_raw": text,
                    "chars_raw": len(text.strip()),
                    "quality_score": text_quality_score(text),
                    "extract_method": "digital",
                }
            )
    return pages


def run_ocrmypdf(input_pdf: Path, output_pdf: Path, language: str = "eng") -> bool:
    if not has_command("ocrmypdf"):
        return False
    cmd = [
        "ocrmypdf",
        "--force-ocr",
        "--skip-text",
        "--optimize",
        "0",
        "-l",
        language,
        str(input_pdf),
        str(output_pdf),
    ]
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
        return proc.returncode == 0
    except Exception:
        return False


def extract_pdf_with_ocr_fallback(
    pdf_path: Path,
    ocr_threshold: float = 0.15,
    ocr_language: str = "eng",
    allow_ocr: bool = True,
) -> list[dict[str, Any]]:
    """Extract PDF text and selectively replace low-quality pages with OCR output."""
    pages = extract_pdf_digital(pdf_path)
    low_quality = [p for p in pages if p["quality_score"] < ocr_threshold]
    if not low_quality or not allow_ocr:
        return pages

    with tempfile.TemporaryDirectory(prefix="c64_ocr_") as tmpdir:
        ocr_pdf = Path(tmpdir) / "ocr_output.pdf"
        if not run_ocrmypdf(pdf_path, ocr_pdf, language=ocr_language):
            return pages
        try:
            ocr_pages = extract_pdf_digital(ocr_pdf)
        except Exception:
            return pages

    by_page = {p["page_number"]: p for p in ocr_pages}
    improved: list[dict[str, Any]] = []
    for original in pages:
        replacement = by_page.get(original["page_number"])
        if replacement is None:
            improved.append(original)
            continue
        better = replacement["quality_score"] > (original["quality_score"] + 0.05)
        if better:
            replacement["extract_method"] = "ocr"
            improved.append(replacement)
        else:
            improved.append(original)
    return improved


def extract_html_text(path: Path) -> str:
    html = read_text(path)
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator="\n")
    return text


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _header_footer_candidates(lines: list[str]) -> list[str]:
    clean = [ln.strip() for ln in lines if 4 <= len(ln.strip()) <= 120]
    return clean


def remove_repeated_headers_footers(doc_df: pd.DataFrame) -> pd.DataFrame:
    """Remove recurrent page headers/footers within one document group."""
    if doc_df.empty:
        return doc_df

    first_lines: list[str] = []
    last_lines: list[str] = []
    for text in doc_df["text_normalized"].tolist():
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            continue
        first_lines.extend(_header_footer_candidates(lines[:2]))
        last_lines.extend(_header_footer_candidates(lines[-2:]))

    n_pages = max(1, len(doc_df))
    threshold = max(3, int(0.35 * n_pages))
    first_counts = pd.Series(first_lines).value_counts() if first_lines else pd.Series(dtype=int)
    last_counts = pd.Series(last_lines).value_counts() if last_lines else pd.Series(dtype=int)
    remove_first = set(first_counts[first_counts >= threshold].index.tolist())
    remove_last = set(last_counts[last_counts >= threshold].index.tolist())

    def _clean_page(text: str) -> str:
        lines = [ln for ln in text.splitlines()]
        while lines and lines[0].strip() in remove_first:
            lines.pop(0)
        while lines and lines[-1].strip() in remove_last:
            lines.pop()
        return "\n".join(lines).strip()

    out = doc_df.copy()
    out["text_normalized"] = out["text_normalized"].map(_clean_page)
    out["chars_normalized"] = out["text_normalized"].map(len)
    return out


def normalized_for_dedup(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_manifest(source_dir: Path, manifest_path: Path) -> pd.DataFrame:
    files = list_source_files(source_dir)
    rows: list[dict[str, Any]] = []
    for f in files:
        rel = f.relative_to(source_dir).as_posix()
        sha = file_sha256(f)
        stem = re.sub(r"[^a-z0-9]+", "-", f.stem.lower()).strip("-")
        doc_id = f"{stem}-{sha[:10]}"
        rows.append(
            {
                "doc_id": doc_id,
                "source_file": rel,
                "absolute_path": str(f.resolve()),
                "extension": f.suffix.lower(),
                "size_bytes": f.stat().st_size,
                "sha256": sha,
            }
        )

    df = pd.DataFrame(rows)
    ensure_parent(manifest_path)
    df.to_parquet(manifest_path, index=False)
    log(f"manifest rows={len(df)} -> {manifest_path}")
    return df


def stage_extract(
    manifest_path: Path,
    extracted_path: Path,
    source_dir: Path,
    allow_ocr: bool,
    ocr_threshold: float,
    ocr_language: str,
) -> pd.DataFrame:
    """Extract text pages from source files declared in the manifest."""
    manifest = pd.read_parquet(manifest_path)
    rows: list[dict[str, Any]] = []

    for rec in manifest.to_dict(orient="records"):
        doc_id = rec["doc_id"]
        ext = rec["extension"]
        abs_path = Path(rec["absolute_path"])
        src_rel = rec["source_file"]

        if ext == ".pdf":
            try:
                pages = extract_pdf_with_ocr_fallback(
                    abs_path,
                    ocr_threshold=ocr_threshold,
                    ocr_language=ocr_language,
                    allow_ocr=allow_ocr,
                )
            except Exception as e:
                log(f"extract error on {src_rel}: {e}")
                pages = []
        else:
            try:
                text = extract_html_text(abs_path)
                pages = [
                    {
                        "page_number": 1,
                        "text_raw": text,
                        "chars_raw": len(text.strip()),
                        "quality_score": text_quality_score(text),
                        "extract_method": "html",
                    }
                ]
            except Exception as e:
                log(f"extract html error on {src_rel}: {e}")
                pages = []

        for p in pages:
            rows.append(
                {
                    "id": f"{doc_id}:p{p['page_number']}",
                    "doc_id": doc_id,
                    "source_file": src_rel,
                    "source_path": str((source_dir / src_rel).resolve()),
                    "source_type": ext.lstrip("."),
                    "page_number": p["page_number"],
                    "text_raw": p["text_raw"],
                    "chars_raw": p["chars_raw"],
                    "quality_score": p["quality_score"],
                    "extract_method": p["extract_method"],
                }
            )

    df = pd.DataFrame(rows)
    ensure_parent(extracted_path)
    df.to_parquet(extracted_path, index=False)
    log(f"extracted rows={len(df)} -> {extracted_path}")
    return df


def stage_normalize(extracted_path: Path, normalized_path: Path, min_chars: int) -> pd.DataFrame:
    """Normalize extracted text and flag rows that are usable for training."""
    df = pd.read_parquet(extracted_path)
    if df.empty:
        ensure_parent(normalized_path)
        df.to_parquet(normalized_path, index=False)
        return df

    out = df.copy()
    out["text_normalized"] = out["text_raw"].fillna("").map(normalize_text)
    out["chars_normalized"] = out["text_normalized"].map(len)

    cleaned_groups = []
    for _, g in out.groupby("doc_id", sort=False):
        gg = g.sort_values("page_number")
        gg = remove_repeated_headers_footers(gg)
        cleaned_groups.append(gg)
    out = pd.concat(cleaned_groups, ignore_index=True)

    out["keep_for_training"] = out["chars_normalized"] >= min_chars
    ensure_parent(normalized_path)
    out.to_parquet(normalized_path, index=False)
    log(f"normalized rows={len(out)} -> {normalized_path}")
    return out


@dataclass
class _Candidate:
    id: str
    text: str
    length: int


def stage_dedup(
    normalized_path: Path,
    dedup_path: Path,
    duplicates_path: Path,
    near_dup_threshold: float = 0.985,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run exact and near-duplicate removal while keeping the best-quality page variant."""
    df = pd.read_parquet(normalized_path)
    if df.empty:
        ensure_parent(dedup_path)
        df.to_parquet(dedup_path, index=False)
        ensure_parent(duplicates_path)
        df.to_parquet(duplicates_path, index=False)
        return df, df

    work = df[df["keep_for_training"]].copy()
    work["dedup_text"] = work["text_normalized"].map(normalized_for_dedup)
    work["exact_hash"] = work["dedup_text"].map(lambda t: hashlib.sha1(t.encode("utf-8")).hexdigest())
    work["quality_rank"] = work["quality_score"].fillna(0.0) * 1_000_000 + work["chars_normalized"].fillna(0)

    # Exact dedup: keep best per hash.
    keep_idx = work.sort_values("quality_rank", ascending=False).drop_duplicates("exact_hash").index
    exact_kept = work.loc[keep_idx].copy()

    # Near-dedup compares only within small candidate buckets to keep runtime bounded.
    near_sorted = exact_kept.sort_values(["quality_rank"], ascending=False)
    kept_rows: list[pd.Series] = []
    duplicates: list[dict[str, Any]] = []
    buckets: dict[tuple[int, str], list[_Candidate]] = {}

    for _, row in near_sorted.iterrows():
        text = row["dedup_text"]
        txt_len = len(text)
        len_bucket = txt_len // 400
        prefix = text[:32]
        key_candidates = []
        for lb in (len_bucket - 1, len_bucket, len_bucket + 1):
            key_candidates.extend(buckets.get((lb, prefix), []))

        duplicate_of = None
        for cand in key_candidates:
            ratio = SequenceMatcher(None, text[:5000], cand.text[:5000]).ratio()
            if ratio >= near_dup_threshold:
                duplicate_of = cand.id
                break

        if duplicate_of is not None:
            duplicates.append(
                {
                    "id": row["id"],
                    "doc_id": row["doc_id"],
                    "source_file": row["source_file"],
                    "duplicate_of": duplicate_of,
                }
            )
            continue

        kept_rows.append(row)
        buckets.setdefault((len_bucket, prefix), []).append(
            _Candidate(id=row["id"], text=text, length=txt_len)
        )

    dedup_df = pd.DataFrame(kept_rows).drop(columns=["dedup_text", "exact_hash", "quality_rank"], errors="ignore")
    duplicates_df = pd.DataFrame(duplicates)

    ensure_parent(dedup_path)
    dedup_df.to_parquet(dedup_path, index=False)
    ensure_parent(duplicates_path)
    duplicates_df.to_parquet(duplicates_path, index=False)
    log(f"dedup kept={len(dedup_df)} removed={len(duplicates_df)}")
    return dedup_df, duplicates_df


def assign_doc_splits(doc_ids: list[str], seed: int, train_ratio: float, val_ratio: float) -> dict[str, str]:
    """Assign stable document-level train/validation/test splits."""
    doc_ids = sorted(set(doc_ids))
    rng = random.Random(seed)
    rng.shuffle(doc_ids)

    n = len(doc_ids)
    if n == 0:
        return {}
    if n == 1:
        return {doc_ids[0]: "train"}
    if n == 2:
        return {doc_ids[0]: "train", doc_ids[1]: "test"}

    # For n >= 3 ensure all splits are represented.
    n_val = max(1, int(n * val_ratio))
    n_test = max(1, int(n * (1.0 - train_ratio - val_ratio)))
    n_train = n - n_val - n_test
    if n_train < 1:
        n_train = 1
        if n_val > n_test:
            n_val -= 1
        else:
            n_test -= 1

    split_map: dict[str, str] = {}
    for i, doc_id in enumerate(doc_ids):
        if i < n_train:
            split_map[doc_id] = "train"
        elif i < n_train + n_val:
            split_map[doc_id] = "validation"
        else:
            split_map[doc_id] = "test"
    return split_map


def token_chunks(
    token_ids: list[int],
    block_size: int,
    stride: int,
    min_chunk_tokens: int,
) -> list[list[int]]:
    """Create overlapping token windows for DAPT training."""
    if block_size <= 0:
        raise ValueError("block_size must be > 0")
    step = block_size - stride
    if step <= 0:
        raise ValueError("block_size must be greater than stride")

    chunks: list[list[int]] = []
    for start in range(0, len(token_ids), step):
        chunk = token_ids[start : start + block_size]
        if not chunk:
            break
        if len(chunk) < min_chunk_tokens and start != 0:
            break
        chunks.append(chunk)
    return chunks


def stage_build_dapt(
    dedup_path: Path,
    dapt_dir: Path,
    model_path: Path,
    block_size: int,
    stride: int,
    min_chunk_tokens: int,
    seed: int,
) -> pd.DataFrame:
    """Build DAPT chunks and save split parquet files."""
    df = pd.read_parquet(dedup_path)
    if df.empty:
        dapt_dir.mkdir(parents=True, exist_ok=True)
        empty = pd.DataFrame(columns=["id", "doc_id", "source_file", "page_start", "page_end", "text", "token_count", "split"])
        for split in ("train", "validation", "test"):
            empty.to_parquet(dapt_dir / f"{split}.parquet", index=False)
        return empty

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    docs = (
        df.sort_values(["doc_id", "page_number"])
        .groupby("doc_id", as_index=False)
        .agg(
            text=("text_normalized", lambda x: "\n\n".join(x)),
            source_file=("source_file", "first"),
            page_start=("page_number", "min"),
            page_end=("page_number", "max"),
        )
    )

    split_map = assign_doc_splits(
        docs["doc_id"].tolist(),
        seed=seed,
        train_ratio=0.8,
        val_ratio=0.1,
    )

    rows: list[dict[str, Any]] = []
    for doc in docs.to_dict(orient="records"):
        token_ids = tokenizer(doc["text"], add_special_tokens=False)["input_ids"]
        chunks = token_chunks(token_ids, block_size=block_size, stride=stride, min_chunk_tokens=min_chunk_tokens)
        for i, chunk in enumerate(chunks):
            text = tokenizer.decode(chunk, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            rows.append(
                {
                    "id": f"{doc['doc_id']}:chunk-{i:05d}",
                    "doc_id": doc["doc_id"],
                    "source_file": doc["source_file"],
                    "page_start": int(doc["page_start"]),
                    "page_end": int(doc["page_end"]),
                    "text": text,
                    "token_count": len(chunk),
                    "split": split_map[doc["doc_id"]],
                }
            )

    out = pd.DataFrame(rows)
    dapt_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        split_df = out[out["split"] == split].copy()
        split_df.to_parquet(dapt_dir / f"{split}.parquet", index=False)
    log(
        "dapt chunks "
        f"train={len(out[out['split']=='train'])} "
        f"val={len(out[out['split']=='validation'])} "
        f"test={len(out[out['split']=='test'])}"
    )
    return out


def split_sentences(text: str) -> list[str]:
    """Split text into coarse sentences for lightweight extractive summaries."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def extractive_summary(text: str, max_sentences: int = 4, max_chars: int = 900) -> str:
    """Produce a compact bullet summary directly from excerpt sentences."""
    sentences = split_sentences(text)
    selected: list[str] = []
    running = 0
    for sent in sentences:
        if len(selected) >= max_sentences:
            break
        if running + len(sent) > max_chars and selected:
            break
        selected.append(sent)
        running += len(sent)
    if not selected:
        selected = [text[: min(len(text), 300)].strip()]
    return "\n".join(f"- {s}" for s in selected if s)


def stage_build_sft(
    dedup_path: Path,
    sft_dir: Path,
    seed: int,
    max_examples_per_page: int = 2,
) -> pd.DataFrame:
    """Build SFT JSONL examples from deduplicated normalized pages."""
    df = pd.read_parquet(dedup_path)
    if df.empty:
        sft_dir.mkdir(parents=True, exist_ok=True)
        for split in ("train", "validation", "test"):
            (sft_dir / f"{split}.jsonl").write_text("", encoding="utf-8")
        return pd.DataFrame(columns=["id", "doc_id", "messages", "source_refs", "quality_score", "split"])

    split_map = assign_doc_splits(df["doc_id"].tolist(), seed=seed, train_ratio=0.8, val_ratio=0.1)
    system_msg = C64_SYSTEM_PROMPT

    rows: list[dict[str, Any]] = []
    for rec in df.sort_values(["doc_id", "page_number"]).to_dict(orient="records"):
        text = rec["text_normalized"]
        if len(text) < 350:
            continue
        excerpt = text[:1600]
        summary = extractive_summary(excerpt)
        refs = [{"source_file": rec["source_file"], "page_number": int(rec["page_number"])}]

        # Example 1: explanation-style
        rows.append(
            {
                "id": f"{rec['id']}:sft-1",
                "doc_id": rec["doc_id"],
                "messages": [
                    {"role": "system", "content": system_msg},
                    {
                        "role": "user",
                        "content": (
                            "Explain this Commodore 64 reference excerpt in practical terms:\n\n"
                            f"{excerpt}"
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": (
                            "Here are the key technical points:\n"
                            f"{summary}"
                        ),
                    },
                ],
                "source_refs": refs,
                "quality_score": rec["quality_score"],
                "split": split_map[rec["doc_id"]],
            }
        )

        # Example 2: factual extraction if memory addresses appear.
        if max_examples_per_page > 1:
            addresses = sorted(set(re.findall(r"\$[0-9A-Fa-f]{4}", excerpt)))
            if addresses:
                mention = ", ".join(addresses[:8])
                rows.append(
                    {
                        "id": f"{rec['id']}:sft-2",
                        "doc_id": rec["doc_id"],
                        "messages": [
                            {"role": "system", "content": system_msg},
                            {
                                "role": "user",
                                "content": (
                                    "From this C64 excerpt, list the memory addresses mentioned "
                                    "and what they are used for:\n\n"
                                    f"{excerpt}"
                                ),
                            },
                            {
                                "role": "assistant",
                                "content": (
                                    f"Memory addresses explicitly mentioned: {mention}\n\n"
                                    "Based on the excerpt, these addresses are referenced in the "
                                    "technical context described there."
                                ),
                            },
                        ],
                        "source_refs": refs,
                        "quality_score": rec["quality_score"],
                        "split": split_map[rec["doc_id"]],
                    }
                )

    out = pd.DataFrame(rows)
    sft_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        split_df = out[out["split"] == split].copy()
        out_path = sft_dir / f"{split}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for rec in split_df.to_dict(orient="records"):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log(
        "sft examples "
        f"train={len(out[out['split']=='train'])} "
        f"val={len(out[out['split']=='validation'])} "
        f"test={len(out[out['split']=='test'])}"
    )
    return out


def stage_validate(
    extracted_path: Path,
    dedup_path: Path,
    dapt_dir: Path,
    sft_dir: Path,
    report_path: Path,
) -> dict[str, Any]:
    """Generate dataset quality checks and a machine-readable validation report."""
    report: dict[str, Any] = {
        "ok": True,
        "checks": {},
    }

    extracted = pd.read_parquet(extracted_path) if extracted_path.exists() else pd.DataFrame()
    dedup = pd.read_parquet(dedup_path) if dedup_path.exists() else pd.DataFrame()
    dapt_train = pd.read_parquet(dapt_dir / "train.parquet") if (dapt_dir / "train.parquet").exists() else pd.DataFrame()
    dapt_val = pd.read_parquet(dapt_dir / "validation.parquet") if (dapt_dir / "validation.parquet").exists() else pd.DataFrame()
    dapt_test = pd.read_parquet(dapt_dir / "test.parquet") if (dapt_dir / "test.parquet").exists() else pd.DataFrame()

    sft_counts: dict[str, int] = {}
    for split in ("train", "validation", "test"):
        p = sft_dir / f"{split}.jsonl"
        if not p.exists():
            sft_counts[split] = 0
            continue
        with p.open("r", encoding="utf-8") as f:
            sft_counts[split] = sum(1 for _ in f)

    total_pages = int(len(extracted))
    pages_with_text = int((extracted["chars_raw"] > 0).sum()) if not extracted.empty else 0
    coverage = float(pages_with_text / total_pages) if total_pages else 0.0
    dapt_total = int(len(dapt_train) + len(dapt_val) + len(dapt_test))
    sft_total = int(sum(sft_counts.values()))

    report["checks"] = {
        "total_pages": total_pages,
        "pages_with_text": pages_with_text,
        "coverage_ratio": round(coverage, 4),
        "dedup_pages": int(len(dedup)),
        "dapt_chunks": {
            "train": int(len(dapt_train)),
            "validation": int(len(dapt_val)),
            "test": int(len(dapt_test)),
            "total": dapt_total,
        },
        "sft_examples": {
            **sft_counts,
            "total": sft_total,
        },
    }

    # Soft gates: keep report explicit but do not block execution with hard failures.
    warnings = []
    if coverage < 0.80:
        warnings.append("Low extraction coverage (<80%). OCR tooling may be missing or failing.")
    if dapt_total < 100:
        warnings.append("Low DAPT chunk count (<100).")
    if sft_total < 50:
        warnings.append("Low SFT example count (<50).")
    report["warnings"] = warnings
    report["ok"] = len(warnings) == 0

    write_json(report_path, report)
    log(f"validation report -> {report_path}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C64 data preparation pipeline.")
    parser.add_argument(
        "--stage",
        required=True,
        choices=["manifest", "extract", "normalize", "dedup", "build_dapt", "build_sft", "validate", "all"],
        help="Pipeline stage to run.",
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--allow-ocr", action="store_true", help="Enable OCR fallback for low-quality PDF pages.")
    parser.add_argument("--ocr-threshold", type=float, default=0.15)
    parser.add_argument("--ocr-language", type=str, default="eng")

    parser.add_argument("--min-chars", type=int, default=120)
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--min-chunk-tokens", type=int, default=512)
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    """Dispatch a single pipeline stage or the complete pipeline."""
    if args.stage in ("manifest", "all"):
        build_manifest(args.source_dir, MANIFEST_PATH)

    if args.stage in ("extract", "all"):
        stage_extract(
            manifest_path=MANIFEST_PATH,
            extracted_path=EXTRACTED_PATH,
            source_dir=args.source_dir,
            allow_ocr=args.allow_ocr,
            ocr_threshold=args.ocr_threshold,
            ocr_language=args.ocr_language,
        )

    if args.stage in ("normalize", "all"):
        stage_normalize(EXTRACTED_PATH, NORMALIZED_PATH, min_chars=args.min_chars)

    if args.stage in ("dedup", "all"):
        stage_dedup(NORMALIZED_PATH, DEDUP_PATH, DUPLICATES_PATH)

    if args.stage in ("build_dapt", "all"):
        stage_build_dapt(
            dedup_path=DEDUP_PATH,
            dapt_dir=PROCESSED_DIR / "dapt",
            model_path=args.model_path,
            block_size=args.block_size,
            stride=args.stride,
            min_chunk_tokens=args.min_chunk_tokens,
            seed=args.seed,
        )

    if args.stage in ("build_sft", "all"):
        stage_build_sft(
            dedup_path=DEDUP_PATH,
            sft_dir=PROCESSED_DIR / "sft",
            seed=args.seed,
        )

    if args.stage in ("validate", "all"):
        stage_validate(
            extracted_path=EXTRACTED_PATH,
            dedup_path=DEDUP_PATH,
            dapt_dir=PROCESSED_DIR / "dapt",
            sft_dir=PROCESSED_DIR / "sft",
            report_path=VALIDATION_REPORT_PATH,
        )


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
