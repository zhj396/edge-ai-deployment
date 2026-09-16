"""Tests for the staged-training retry/resume state machine (train/train_staged).

The module under test is loaded by file path with importlib because
``train/train_staged`` is a self-contained script directory whose ``utils.py``
would shadow the repo-root ``utils`` package if the directory were put on
``sys.path``. It imports only numpy/torch/yaml — this file keeps the default
suite free of ``ultralytics`` (see test_consistency.py invariant), so the
resume branch of ``trainer.train_stage`` itself stays opt-in/untested here.

Import side effects (``setup_logging`` creates ``logs/`` relative to CWD) are
confined to a tmp dir via ``monkeypatch.chdir``.
"""
import importlib.util
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TRAIN_STAGED_UTILS = ROOT / "train" / "train_staged" / "utils.py"


@pytest.fixture()
def tsu(tmp_path, monkeypatch):
    """Load train_staged/utils.py as an isolated module with CWD in tmp_path."""
    monkeypatch.chdir(tmp_path)
    spec = importlib.util.spec_from_file_location("train_staged_utils", TRAIN_STAGED_UTILS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # Keep retry tests instant: safe_run sleeps 2**i seconds between attempts.
    monkeypatch.setattr(mod.time, "sleep", lambda _s: None)
    return mod


def _touch(path: Path, mtime: float):
    """Create/rewrite ``path`` with an explicit mtime (deterministic comparisons)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"ckpt")
    os.utime(path, (mtime, mtime))


# ---------------------------------------------------------------------------
# safe_run: on_retry hook
# ---------------------------------------------------------------------------
def test_safe_run_first_attempt_success_skips_on_retry(tsu):
    calls = []
    result = tsu.safe_run(lambda: "ok", name="t", retries=3, on_retry=calls.append)
    assert result == "ok"
    assert calls == []


def test_safe_run_calls_on_retry_between_attempts(tsu):
    calls = []
    attempts = {"n": 0}

    def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError("boom")
        return "ok"

    result = tsu.safe_run(flaky, name="t", retries=3, on_retry=calls.append)
    assert result == "ok"
    assert calls == [1, 2]  # 1-based number of the attempt that just failed


def test_safe_run_exhausted_raises_and_skips_final_on_retry(tsu):
    calls = []

    def always_fail():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        tsu.safe_run(always_fail, name="t", retries=3, on_retry=calls.append)
    # on_retry fires only *between* attempts: 3 attempts -> 2 retries.
    assert calls == [1, 2]


def test_safe_run_remains_backward_compatible_without_on_retry(tsu):
    assert tsu.safe_run(lambda: 42, name="t") == 42


# ---------------------------------------------------------------------------
# checkpoint_retry_runner: resume decision from checkpoint mtime
# ---------------------------------------------------------------------------
def test_runner_fresh_success_never_resumes(tsu, tmp_path):
    ckpt = tmp_path / "weights" / "last.pt"
    seen = []

    def run_fn(resume):
        seen.append(resume)
        return "ok"

    run, _ = tsu.checkpoint_retry_runner(run_fn, ckpt, name="Stage A")
    assert run() == "ok"
    assert seen == [False]


def test_runner_resumes_when_failed_attempt_checkpointed(tsu, tmp_path):
    ckpt = tmp_path / "weights" / "last.pt"
    seen = []

    def run_fn(resume):
        seen.append(resume)
        if len(seen) == 1:
            # Simulate: trained 43 epochs (checkpoint saved), then crashed.
            _touch(ckpt, mtime=1000.0)
            raise RuntimeError("BN crash")
        return "resumed-ok"

    run, on_retry = tsu.checkpoint_retry_runner(run_fn, ckpt, name="Stage A")
    with pytest.raises(RuntimeError):
        run()
    on_retry(1)
    assert run() == "resumed-ok"
    assert seen == [False, True]


def test_runner_ignores_stale_checkpoint_from_previous_run(tsu, tmp_path):
    ckpt = tmp_path / "weights" / "last.pt"
    _touch(ckpt, mtime=500.0)  # left behind by an earlier (exist_ok) run
    seen = []

    def run_fn(resume):
        seen.append(resume)
        if len(seen) == 1:
            raise RuntimeError("crashed before any save")
        return "fresh-ok"

    run, on_retry = tsu.checkpoint_retry_runner(run_fn, ckpt, name="Stage A")
    with pytest.raises(RuntimeError):
        run()
    on_retry(1)
    assert run() == "fresh-ok"
    # Stale checkpoint mtime == baseline -> must NOT be mistaken for crash state.
    assert seen == [False, False]


def test_runner_restarts_fresh_when_failure_saved_nothing(tsu, tmp_path):
    ckpt = tmp_path / "weights" / "last.pt"
    seen = []

    def run_fn(resume):
        seen.append(resume)
        if len(seen) == 1:
            raise RuntimeError("instant config error")
        return "fresh-ok"

    run, on_retry = tsu.checkpoint_retry_runner(run_fn, ckpt)
    with pytest.raises(RuntimeError):
        run()
    on_retry(1)
    assert run() == "fresh-ok"
    assert seen == [False, False]
    assert not ckpt.exists()


def test_runner_resumes_again_after_second_failure(tsu, tmp_path):
    ckpt = tmp_path / "weights" / "last.pt"
    seen = []

    def run_fn(resume):
        seen.append(resume)
        if len(seen) <= 2:
            # Each failed attempt advances the checkpoint (further epochs done).
            _touch(ckpt, mtime=1000.0 + len(seen))
            raise RuntimeError("crash again")
        return "done"

    run, on_retry = tsu.checkpoint_retry_runner(run_fn, ckpt, name="Stage A")
    for attempt in (1, 2):
        with pytest.raises(RuntimeError):
            run()
        on_retry(attempt)
    assert run() == "done"
    assert seen == [False, True, True]


def test_runner_integrates_with_safe_run(tsu, tmp_path):
    """End-to-end: safe_run + runner = crash-with-checkpoint then resume."""
    ckpt = tmp_path / "weights" / "last.pt"
    seen = []

    def run_fn(resume):
        seen.append(resume)
        if not resume:
            _touch(ckpt, mtime=1000.0)  # epoch saved before the crash
            raise RuntimeError("mid-stage crash")
        return "best.pt"

    run, on_retry = tsu.checkpoint_retry_runner(run_fn, ckpt, name="Stage B")
    result = tsu.safe_run(run, name="Stage B", retries=2, on_retry=on_retry)
    assert result == "best.pt"
    assert seen == [False, True]
