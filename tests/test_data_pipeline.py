"""Unit tests for data-pipeline helpers and split/chunk behavior."""

from scripts.data_pipeline import (
    assistant_has_think_tags,
    assign_doc_splits,
    compact_bullets,
    collect_sft_thinking_stats,
    extract_think_content,
    format_assistant_with_think,
    is_low_signal_sft_text,
    normalize_text,
    token_chunks,
)


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


def test_format_assistant_with_think_wraps_reasoning_and_final_answer():
    out = format_assistant_with_think("trace register map", "Final answer.")
    assert out.startswith("[THINK]trace register map[/THINK]")
    assert out.endswith("Final answer.")
    assert assistant_has_think_tags(out)


def test_assistant_has_think_tags_requires_balanced_markers():
    assert assistant_has_think_tags("[THINK]x[/THINK] body")
    assert not assistant_has_think_tags("missing tags")
    assert not assistant_has_think_tags("[/THINK][THINK]")
    assert not assistant_has_think_tags("[THINK][/THINK] no body")


def test_extract_think_content_normalizes_whitespace():
    out = extract_think_content("[THINK]  a   b\nc [/THINK]\nanswer")
    assert out == "a b c"


def test_compact_bullets_limits_items():
    summary = "- one\n- two\n- three\n- four\n"
    assert compact_bullets(summary, max_items=2) == "- one\n- two"


def test_is_low_signal_sft_text_filters_boilerplate_but_keeps_technical_text():
    low_signal = "TABLE OF CONTENTS " + "1 2 3 4 5 " * 80
    technical = (
        "The VIC-II raster interrupt uses $D012 for line compare and $D011 high bit control. "
        "For stable timing, configure CIA interrupt masks and acknowledge IRQ sources correctly. "
        "This section explains how BASIC and ML routines coordinate display updates without jitter. "
    ) * 4
    assert is_low_signal_sft_text(low_signal)
    assert not is_low_signal_sft_text(technical)


def test_collect_sft_thinking_stats_counts_assistant_think_tags(tmp_path):
    sft_dir = tmp_path / "sft"
    sft_dir.mkdir()
    train = sft_dir / "train.jsonl"
    train.write_text(
        "\n".join(
            [
                '{"messages":[{"role":"assistant","content":"[THINK]a[/THINK]\\nfinal"}]}',
                '{"messages":[{"role":"assistant","content":"final only"}]}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    for split in ("validation", "test"):
        (sft_dir / f"{split}.jsonl").write_text("", encoding="utf-8")

    stats = collect_sft_thinking_stats(sft_dir)
    assert stats["assistant_messages_total"] == 2
    assert stats["assistant_with_think_total"] == 1
    assert stats["assistant_with_think_ratio"] == 0.5
    assert stats["unique_think_texts"] == 1
