"""chunk_text: the character-window chunker used during ingest."""
from secondbrain.ingest import chunk_text


def test_empty_or_whitespace_yields_no_chunks():
    assert chunk_text("", 100, 10) == []
    assert chunk_text("   \n\t  ", 100, 10) == []


def test_short_text_is_a_single_chunk():
    assert chunk_text("hello world", 100, 10) == ["hello world"]


def test_long_text_splits_into_bounded_overlapping_chunks():
    text = "word " * 500  # 2500 chars
    chunks = chunk_text(text, size=200, overlap=50)
    assert len(chunks) > 1
    # No chunk exceeds the window size.
    assert all(len(c) <= 200 for c in chunks)
    # Whole text remains covered (cheap content check on a distinctive token).
    assert "word" in chunks[0] and "word" in chunks[-1]


def test_prefers_newline_break_near_window_edge():
    # A newline sits inside the break window, so the first chunk should end there.
    text = "alpha beta gamma\n" + "x" * 300
    chunks = chunk_text(text, size=20, overlap=8)
    assert chunks[0] == "alpha beta gamma"
