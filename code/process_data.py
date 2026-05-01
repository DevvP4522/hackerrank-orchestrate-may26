import json
import re
import warnings
from pathlib import Path
from typing import Dict, List, Set, Tuple

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning


def clean_html(content: str) -> str:
    """
    Remove HTML tags and noisy sections, then normalize whitespace.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(content, "html.parser")

    for tag_name in ("script", "style", "nav", "header", "footer"):
        for tag in soup.find_all(tag_name):
            tag.decompose()

    text = soup.get_text(separator=" ")
    # Normalize whitespace: collapse multiple spaces/newlines/tabs into single spaces.
    text = re.sub(r"\s+", " ", text).strip()
    return text


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    """
    Split text into word-based chunks with overlap.
    """
    words = text.split()
    if not words:
        return []

    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be < chunk_size")

    stride = chunk_size - overlap
    chunks: List[str] = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunk = " ".join(chunk_words).strip()
        if chunk:
            chunks.append(chunk)
        start += stride

    return chunks


def infer_product_area(filename: str) -> str:
    """
    Infer product area from filename (lowercased) using substring rules.
    """
    name = filename.lower()

    if "account" in name:
        return "account"
    if "billing" in name or "payment" in name:
        return "billing"
    if "fraud" in name:
        return "fraud"
    if "api" in name:
        return "api"
    if "assessment" in name:
        return "assessment"
    return "general"


def _should_keep_chunk(chunk: str, min_words: int = 10) -> bool:
    if not chunk:
        return False
    words = chunk.split()
    if len(words) < min_words:
        return False
    # "Mostly empty" guard: if it has too little alphanumeric content, skip it.
    alnum = re.sub(r"[^a-zA-Z0-9]+", "", chunk)
    if not alnum:
        return False
    return True


def process_data() -> None:
    project_root = Path(__file__).resolve().parent.parent
    data_root = project_root / "data"
    out_path = project_root / "processed_chunks.json"

    sources: List[Tuple[str, Path]] = [
        ("hackerrank", data_root / "hackerrank"),
        ("claude", data_root / "claude"),
        ("visa", data_root / "visa"),
    ]

    chunk_size_words = 400
    overlap_words = 50
    min_file_chars = 50

    total_files = 0
    unreadable_files = 0
    kept_files = 0
    raw_chars_total = 0

    deduped_chunks: List[Dict[str, str]] = []
    seen_text_chunks: Set[str] = set()

    for source, source_dir in sources:
        if not source_dir.exists() or not source_dir.is_dir():
            print(f"[warn] Missing source directory: {source_dir} (skipping)")
            continue

        file_paths = sorted([p for p in source_dir.rglob("*") if p.is_file()])
        print(f"[info] Scanning {source_dir} ({len(file_paths)} files)")

        for path in file_paths:
            total_files += 1

            try:
                # Read safely, handling encoding errors.
                content = path.read_text(encoding="utf-8", errors="replace")
            except Exception as exc:
                unreadable_files += 1
                print(f"[warn] Could not read file: {path} ({exc})")
                continue

            if len(content) < min_file_chars:
                continue

            raw_chars_total += len(content)
            kept_files += 1

            cleaned = clean_html(content)
            if not cleaned:
                continue

            product_area = infer_product_area(path.name)
            word_chunks = chunk_text(
                cleaned, chunk_size=chunk_size_words, overlap=overlap_words
            )

            for chunk in word_chunks:
                if not _should_keep_chunk(chunk):
                    continue

                # Remove duplicate chunks (text-only dedupe).
                if chunk in seen_text_chunks:
                    continue
                seen_text_chunks.add(chunk)

                deduped_chunks.append(
                    {
                        "text": chunk,
                        "source": source,
                        "product_area": product_area,
                    }
                )

    # Deterministic output: stable ordering by insertion is deterministic due to sorted file traversal.
    # (However, dedup affects which are inserted; still deterministic given deterministic traversal.)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(deduped_chunks, f, ensure_ascii=False, indent=2)

    print(f"[done] Processed files: total={total_files}, kept={kept_files}, unreadable={unreadable_files}")
    print(f"[done] Raw chars processed (kept files only): {raw_chars_total}")
    print(f"[done] Total chunks written: {len(deduped_chunks)}")
    print(f"[done] Output path: {out_path}")


if __name__ == "__main__":
    process_data()
