"""Unit tests for src/engine.py's fixed-batch chunking helper (pure logic, no model)."""
from src.engine import _batch_chunks


def test_batch_chunks_even_split():
    assert _batch_chunks([1, 2, 3, 4], 2) == [([1, 2], 2), ([3, 4], 2)]


def test_batch_chunks_short_last_is_padded_and_reported():
    chunks = _batch_chunks([1, 2, 3], 2)
    assert chunks[0] == ([1, 2], 2)
    chunk, keep = chunks[1]
    assert chunk == [3, 3]  # padded with repeats of the chunk's first item
    assert keep == 1


def test_batch_chunks_batch_of_one():
    assert _batch_chunks([7, 8], 1) == [([7], 1), ([8], 1)]


def test_batch_chunks_empty_input_yields_no_chunks():
    assert _batch_chunks([], 4) == []
