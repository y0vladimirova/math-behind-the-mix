"""Tests for audio loading, key label normalisation and the dataset reader."""

import numpy as np
import pytest
import soundfile as sf

from djmath.datasets import load_giantsteps_key
from djmath.key import normalize_key_label
from djmath.signals import load_audio, magnitude_spectrum, sine_wave


@pytest.mark.parametrize(
    "text, label",
    [
        ("C major", "C major"),
        ("Eb minor", "D# minor"),
        ("Bb major", "A# major"),
        ("  F# Minor\n", "F# minor"),
    ],
)
def test_normalize_key_label(text, label):
    assert normalize_key_label(text) == label


@pytest.mark.parametrize("text", ["", "C", "H major", "C dorian", "C major extra"])
def test_normalize_key_label_rejects_invalid_text(text):
    with pytest.raises(ValueError):
        normalize_key_label(text)


def test_load_audio_converts_stereo_44k_to_mono_22k(tmp_path):
    tone = sine_wave(440, duration=1.0, sample_rate=44_100, amplitude=0.5)
    stereo = np.column_stack([tone, tone])
    path = tmp_path / "tone.wav"
    sf.write(path, stereo, 44_100)

    x = load_audio(path, sample_rate=22_050)
    assert x.ndim == 1
    assert len(x) == 22_050
    freqs, amps = magnitude_spectrum(x, sample_rate=22_050)
    assert freqs[np.argmax(amps)] == pytest.approx(440)
    assert amps.max() == pytest.approx(0.5, abs=0.01)


def test_resampling_removes_frequencies_above_the_new_nyquist(tmp_path):
    # 15 kHz is fine at 44.1 kHz but above Nyquist (11 025 Hz) at 22.05 kHz
    path = tmp_path / "high.wav"
    sf.write(path, sine_wave(15_000, duration=1.0, sample_rate=44_100), 44_100)
    x = load_audio(path, sample_rate=22_050)
    middle = x[2000:-2000]   # skip the filter's short start-up effects at the edges
    assert np.max(np.abs(middle)) < 0.01   # removed instead of aliasing to 7 050 Hz


def make_fake_dataset(root, keys, with_audio):
    (root / "annotations" / "key").mkdir(parents=True)
    (root / "audio").mkdir()
    for name, key in keys.items():
        (root / "annotations" / "key" / f"{name}.LOFI.key").write_text(key)
        if name in with_audio:
            (root / "audio" / f"{name}.LOFI.mp3").write_bytes(b"")


def test_load_giantsteps_key_lists_downloaded_tracks(tmp_path):
    make_fake_dataset(tmp_path, {"1": "Eb minor", "2": "C major", "3": "A minor"},
                      with_audio={"1", "3"})
    tracks = load_giantsteps_key(tmp_path)
    assert [(path.name, key) for path, key in tracks] == [
        ("1.LOFI.mp3", "D# minor"),
        ("3.LOFI.mp3", "A minor"),
    ]


def test_load_giantsteps_key_without_annotations_explains_what_to_do(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_data.py"):
        load_giantsteps_key(tmp_path)


