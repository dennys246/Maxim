"""Piper voice provisioning (#1268 follow-up F1).

The Reachy app's spoken-code pairing speaks through a Piper voice nothing provisioned: without it the announcer fails
silently while the sign-in page says "listen". The voice is now pinned (one HF revision, size + sha256 per file),
verified on download AND when already present, provisioned by ``ensure_tts_voice`` under its own lock, and a device
arms the announcer only once a background provisioner has the voice LOADED (owner decision 2026-10-14: no network at
boot means retry in the background and arm when ready). No test touches the network.
"""

from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path

import pytest

from maxim.models import download as dl

_PIN = "c10ece1aade47bb51c153c893d14e5bf8e5b7117"
_ONNX, _JSON = b"fake-onnx-weights", b'{"audio": {"sample_rate": 22050}}'


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@pytest.fixture()
def fake_voice(monkeypatch, tmp_path):
    """A tiny registered voice and a fake fetch that serves its bytes; counts fetches."""
    files = {"fake.onnx": _ONNX, "fake.onnx.json": _JSON}
    entry = {
        "description": "test voice",
        "size_mb": 1,
        "files": [
            {"url": f"https://example.invalid/{n}", "filename": n, "expected_bytes": len(b), "sha256": _sha(b)}
            for n, b in files.items()
        ],
    }
    monkeypatch.setitem(dl.TTS_MODELS, "fake", entry)
    fetched: list[str] = []
    served = dict(files)

    def fetch(url, dest, **kwargs):
        name = url.rsplit("/", 1)[-1]
        fetched.append(name)
        Path(dest).write_bytes(served[name])

    monkeypatch.setattr("maxim.utils.http.download_to_file", fetch)
    monkeypatch.setenv("MAXIM_DATA_HOME", str(tmp_path / "data"))
    return tmp_path / "tts", fetched, served


def test_every_registered_voice_file_is_pinned_and_hashed():
    for name, info in dl.TTS_MODELS.items():
        for f in info["files"]:
            assert f"/resolve/{_PIN}/" in f["url"], (name, f["url"])
            assert isinstance(f["expected_bytes"], int) and f["expected_bytes"] > 0
            assert re.fullmatch(r"[0-9a-f]{64}", f["sha256"])


def test_a_download_whose_hash_does_not_match_is_rejected(fake_voice):
    tts_dir, _, served = fake_voice
    served["fake.onnx"] = b"x" * len(_ONNX)  # right size, wrong bytes
    dest = tts_dir / "fake.onnx"
    assert not dl.download_file(
        "https://example.invalid/fake.onnx", dest, expected_bytes=len(_ONNX), expected_sha256=_sha(_ONNX)
    )
    assert not dest.exists() and not list(tts_dir.glob("*.partial"))


def test_a_present_file_with_wrong_bytes_is_replaced(fake_voice):
    tts_dir, fetched, _ = fake_voice
    dest = tts_dir / "fake.onnx"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"y" * len(_ONNX))  # e.g. an earlier unverified wget of `main`
    assert dl.download_file(
        "https://example.invalid/fake.onnx", dest, expected_bytes=len(_ONNX), expected_sha256=_sha(_ONNX)
    )
    assert dest.read_bytes() == _ONNX and fetched == ["fake.onnx"]


def test_ensure_tts_voice_provisions_verifies_and_is_idempotent(fake_voice):
    tts_dir, fetched, _ = fake_voice
    assert dl.ensure_tts_voice("fake", auto=True, models_dir=tts_dir)
    assert (tts_dir / "fake.onnx").read_bytes() == _ONNX and (tts_dir / "fake.onnx.json").read_bytes() == _JSON
    assert dl.ensure_tts_voice("fake", auto=True, models_dir=tts_dir)
    assert sorted(fetched) == ["fake.onnx", "fake.onnx.json"]  # the second call verified, fetched nothing
    (tts_dir / "fake.onnx").write_bytes(b"z" * len(_ONNX))  # corrupted in place, same size
    assert dl.ensure_tts_voice("fake", auto=True, models_dir=tts_dir)
    assert (tts_dir / "fake.onnx").read_bytes() == _ONNX
    assert not dl.ensure_tts_voice("no-such-voice", auto=True, models_dir=tts_dir)


def test_ensure_tts_voice_without_consent_downloads_nothing(fake_voice):
    tts_dir, fetched, _ = fake_voice
    assert not dl.ensure_tts_voice("fake", auto=False, interactive=False, models_dir=tts_dir)
    assert fetched == []


def test_a_model_download_in_another_process_does_not_block_the_voice(fake_voice):
    """The voice takes its own lock: a 40 GB GGUF download holding download.lock must not hold up a robot's sign-in."""
    from maxim.utils.paths import data_home
    from maxim.utils.process_lock import file_lock

    tts_dir, _, _ = fake_voice
    with file_lock(data_home() / "util" / "download.lock"):
        assert dl.ensure_tts_voice("fake", auto=True, models_dir=tts_dir)


def test_the_provisioner_arms_only_once_the_voice_loads_and_retries_until_then(monkeypatch, tmp_path):
    from maxim.models.audio import tts

    attempts: list[int] = []

    def ensure(*args, **kwargs):
        attempts.append(1)
        return len(attempts) >= 3  # no network for the first two tries

    loaded: list[int] = []
    monkeypatch.setattr(dl, "ensure_tts_voice", ensure)
    monkeypatch.setattr(tts, "_piper_installed", lambda: True)
    monkeypatch.setattr(tts.TTSEngine, "load", lambda self: loaded.append(1))
    prov = tts.VoiceProvisioner(models_dir=tmp_path, retry_s=0.01, max_retry_s=0.02)
    assert prov.is_ready() is False
    prov.start()
    try:
        for _ in range(300):
            if prov.is_ready():
                break
            time.sleep(0.01)
        assert prov.is_ready() and len(attempts) == 3 and loaded == [1]
    finally:
        prov.stop()


def test_without_piper_installed_the_provisioner_never_arms(monkeypatch, tmp_path):
    from maxim.models.audio import tts

    monkeypatch.setattr(dl, "ensure_tts_voice", lambda *a, **k: True)
    monkeypatch.setattr(tts, "_piper_installed", lambda: False)
    monkeypatch.setattr(tts.TTSEngine, "load", lambda self: None)  # would load: only the piper check refuses
    prov = tts.VoiceProvisioner(models_dir=tmp_path, retry_s=0.01, max_retry_s=0.02)
    prov.start()
    try:
        time.sleep(0.1)
        assert prov.is_ready() is False
    finally:
        prov.stop()


def test_maxim_serve_never_provisions_a_voice():
    """Only a device bootstrap arms an announcer; a desktop `maxim serve` must not download a voice."""
    import ast

    src = Path(__file__).resolve().parents[2] / "src" / "maxim"
    banned = {"ensure_tts_voice", "VoiceProvisioner"}
    for rel in ("console/server.py", "cli.py"):
        tree = ast.parse((src / rel).read_text(encoding="utf-8"))
        used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        used |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        used |= {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
        assert not used & banned, (rel, used & banned)  # code, not docstrings: a doc may name the device path


def test_a_raise_inside_an_attempt_is_a_retry_not_a_dead_thread(monkeypatch, tmp_path):
    """A lock, stat or storage error inside ensure_tts_voice must not kill the provisioner: the device would stay
    paste-only until a restart, the failure this exists to self-heal (three-lens review)."""
    from maxim.models.audio import tts

    calls: list[int] = []

    def ensure(*args, **kwargs):
        calls.append(1)
        if len(calls) < 3:
            raise OSError("lock dir unwritable")
        return True

    monkeypatch.setattr(dl, "ensure_tts_voice", ensure)
    monkeypatch.setattr(tts, "_piper_installed", lambda: True)
    monkeypatch.setattr(tts.TTSEngine, "load", lambda self: None)
    prov = tts.VoiceProvisioner(models_dir=tmp_path, retry_s=0.01, max_retry_s=0.02)
    prov.start()
    try:
        for _ in range(300):
            if prov.is_ready():
                break
            time.sleep(0.01)
        assert prov.is_ready() and len(calls) == 3
    finally:
        prov.stop()
