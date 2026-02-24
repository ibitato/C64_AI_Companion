"""Unit tests for data-pipeline helpers and split/chunk behavior."""

from scripts.data_pipeline import assign_doc_splits, normalize_text, token_chunks


def test_normalize_preserves_technical_symbols_and_case():
    raw = "POKE 53280,0\r\nAddress: $D020\t\t\n\n"
    cleaned = normalize_text(raw)
    assert "POKE" in cleaned
    assert "$D020" in cleaned
    assert "poke" not in cleaned


def test_assign_doc_splits_non_empty_for_three_or_more_docs():
    doc_ids = [f"doc-{i}" for i in range(8)]
    split_map = assign_doc_splits(doc_ids, seed=42, train_ratio=0.8, val_ratio=0.1)
    values = list(split_map.values())
    assert "train" in values
    assert "validation" in values
    assert "test" in values


def test_token_chunks_overlap_and_min_chunk():
    tokens = list(range(5000))
    chunks = token_chunks(tokens, block_size=1024, stride=128, min_chunk_tokens=256)
    assert len(chunks) > 1
    assert len(chunks[0]) == 1024
    # Overlap check between first and second chunk.
    assert chunks[0][-128:] == chunks[1][:128]
