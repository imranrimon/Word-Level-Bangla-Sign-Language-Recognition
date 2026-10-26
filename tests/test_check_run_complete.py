"""tools/check_run_complete.py on synthetic main.py work_dirs."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import check_run_complete as C  # noqa: E402


def _run(tmp_path, name, num_epoch, last_eval):
    wd = tmp_path / name
    wd.mkdir()
    (wd / "config.yaml").write_text(f"phase: train\nnum_epoch: {num_epoch}\n")
    log = "".join(f"[ Mon Oct 20 10:00:00 2026 ] Training epoch: {e}\n"
                  f"[ Mon Oct 20 10:00:01 2026 ] Eval epoch: {e}\n\tTop1: {40 + e:.2f}%\n"
                  for e in range(1, last_eval + 1))
    (wd / "log.txt").write_text(log)
    return wd


def test_complete_run(tmp_path):
    st = C.train_status(_run(tmp_path, "a_seed0", 100, 100))
    assert st["status"] == "COMPLETE" and st["complete"] is True


def test_truncated_run(tmp_path):
    wd = _run(tmp_path, "b_seed0", 100, 25)
    (wd / "b_seed0_model-best.pt").write_bytes(b"")      # a best checkpoint proves nothing
    st = C.train_status(wd)
    assert st["complete"] is False and st["status"] in ("INCOMPLETE", "IN_PROGRESS")


def test_final_checkpoint_counts_as_complete(tmp_path):
    wd = _run(tmp_path, "c_seed0", 100, 90)
    (wd / "c_seed0_model-99.pt").write_bytes(b"")
    assert C.train_status(wd)["complete"] is True


def test_missing_run(tmp_path):
    st = C.train_status(tmp_path / "nope")
    assert st["status"] == "MISSING" and st["complete"] is False


def test_p1_sweep_enumerates_both_studies():
    runs = C.p1_runs()
    groups = {g for g, _ in runs}
    assert groups == {"v1 7-pseed refit", "v1 BdSL headline", "v2 rerun"}
    assert sum(g == "v2 rerun" for g, _ in runs) == 8 * 2 * 3 * 5
    assert sum(g == "v1 7-pseed refit" for g, _ in runs) == (4 * 3 + 3 * 5) * 7 * 2
