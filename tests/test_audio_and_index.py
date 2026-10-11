import json

import numpy as np
import pytest
import soundfile as sf

from ingest import atomic_write_text
from retrieve import check_index_consistency
from stt import load_audio_16k


def write_wav(path, rate, seconds=1.0, channels=1):
    t = np.linspace(0, seconds, int(rate * seconds), endpoint=False)
    tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype("float32")
    data = np.stack([tone] * channels, axis=1) if channels > 1 else tone
    sf.write(str(path), data, rate, subtype="PCM_16")


def test_16k_mono_passes_through(tmp_path):
    write_wav(tmp_path / "a.wav", 16000)
    audio = load_audio_16k(str(tmp_path / "a.wav"))
    assert audio.dtype == np.float32 and audio.ndim == 1 and len(audio) == 16000
    assert 0.4 < np.abs(audio).max() <= 1.0


def test_stereo_is_averaged_to_mono(tmp_path):
    write_wav(tmp_path / "s.wav", 16000, channels=2)
    assert load_audio_16k(str(tmp_path / "s.wav")).ndim == 1


@pytest.mark.parametrize("rate", [8000, 22050, 44100, 48000])
def test_other_sample_rates_are_resampled_to_16k(tmp_path, rate):
    write_wav(tmp_path / "r.wav", rate, seconds=2.0)
    audio = load_audio_16k(str(tmp_path / "r.wav"))
    assert abs(len(audio) - 32000) <= 1


def test_consistency_ok():
    check_index_consistency(3, [{}, {}, {}], {"embedding_model": "m"}, "m")
    check_index_consistency(2, [{}, {}], None, "m")  # no metadata file -> only counts checked


def test_consistency_detects_count_mismatch():
    with pytest.raises(RuntimeError, match="out of sync"):
        check_index_consistency(5, [{}, {}], {"embedding_model": "m"}, "m")


def test_consistency_detects_embedding_model_mismatch():
    with pytest.raises(RuntimeError, match="built with"):
        check_index_consistency(1, [{}], {"embedding_model": "old"}, "new")


def test_atomic_write_creates_file_and_leaves_no_temp(tmp_path):
    target = tmp_path / "chunks.json"
    atomic_write_text(target, json.dumps([1, 2]))
    assert json.loads(target.read_text()) == [1, 2]
    assert [p.name for p in tmp_path.iterdir()] == ["chunks.json"]


def test_atomic_write_failure_keeps_old_file(tmp_path):
    target = tmp_path / "chunks.json"
    target.write_text("OLD")
    with pytest.raises(TypeError):
        atomic_write_text(target, 12345)  # not a string -> write fails
    assert target.read_text() == "OLD"
    assert [p.name for p in tmp_path.iterdir()] == ["chunks.json"]  # temp file cleaned up
