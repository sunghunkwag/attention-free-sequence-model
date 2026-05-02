import subprocess
import sys


def test_dry_run_cpu_succeeds(tmp_path):
    corpus = ("abcdefghijklmnopqrstuvwxyz\n" * 60)
    data_path = tmp_path / "tiny.txt"
    data_path.write_text(corpus, encoding="utf-8")

    cmd = [
        sys.executable,
        "train_language_model.py",
        "--data-path",
        str(data_path),
        "--seq-len",
        "16",
        "--stride",
        "16",
        "--batch-size",
        "4",
        "--d-model",
        "16",
        "--num-layers",
        "1",
        "--epochs",
        "1",
        "--device",
        "cpu",
        "--dry-run",
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert "dry_run_ok" in out.stdout


def test_dry_run_too_small_corpus_raises_clear_error(tmp_path):
    corpus = "short corpus"
    data_path = tmp_path / "small.txt"
    data_path.write_text(corpus, encoding="utf-8")

    cmd = [
        sys.executable,
        "train_language_model.py",
        "--data-path",
        str(data_path),
        "--seq-len",
        "16",
        "--stride",
        "16",
        "--batch-size",
        "1024",
        "--d-model",
        "16",
        "--num-layers",
        "1",
        "--epochs",
        "1",
        "--device",
        "cpu",
        "--dry-run",
    ]
    out = subprocess.run(cmd, capture_output=True, text=True)
    assert out.returncode != 0
    assert "Corpus split has" in (out.stderr + out.stdout)
