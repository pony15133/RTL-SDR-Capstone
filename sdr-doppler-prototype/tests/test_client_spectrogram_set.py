"""The client's labelled pass folders become IQ- and waterfall-model training rows."""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import build_client_spectrogram_set as b  # noqa: E402

NB, ROWS = 1024, 40
BIN = b.DEFAULT_SPAN_HZ / NB


def _snapshot(rng, offset_hz=None, db=14.0):
    m = rng.normal(-6, 2.0, (ROWS, NB)).astype(np.float32)
    m[:, NB // 2] += 8            # DC spike, like the real data
    m[:, NB // 2 + 21] += 7       # fixed local spur (+4.9 kHz)
    if offset_hz is not None:
        c = NB // 2 + int(round(offset_hz / BIN))
        m[:, c - 1:c + 2] += db
    return m


def _make_pass(root, name, freq, parts, rng, t0):
    """parts: [(folder, n_snapshots, has_signal)] in time order; Doppler +10 -> -10 kHz over the pass."""
    total = sum(n for _, n, _ in parts)
    k = 0
    for folder, n, sig in parts:
        d = root / name / folder
        d.mkdir(parents=True)
        for i in range(n):
            frac = k / max(1, total - 1)
            off = 10_000 * np.cos(np.pi * frac) if sig else None
            f = d / f"spectrogram{i}_{freq}.txt"
            np.savetxt(f, _snapshot(rng, off), fmt="%.2f")
            (d / f"Freqtest_{i}_{freq}.png").write_bytes(b"png")
            os.utime(f, (t0 + 1.6 * k, t0 + 1.6 * k))
            k += 1


def test_build_labels_and_splits(tmp_path):
    rng = np.random.default_rng(0)
    aa = tmp_path / "client_pass_recordings"
    _make_pass(aa, "SARAL_pass1_detected", 465988000,
               [("notinsky", 12, False), ("justrisen", 12, True), ("max35", 12, True), ("5to-5", 12, True),
                ("-5to-10", 8, False)], rng, 1_000_000)
    _make_pass(aa, "SARAL_pass2_detected", 465988000,
               [("notinskyto3", 30, False), ("21to62", 12, True), ("61to40", 12, True)], rng, 2_000_000)
    _make_pass(aa, "LILACSAT-2_not_detected", 437200000,
               [("notinskyto7", 12, False), ("8tomax", 12, False)], rng, 3_000_000)
    r = b.build(aa, tmp_path / "out", tmp_path / "work", snapshot_s=0.2)
    s = r["summary"].set_index("pass")
    assert s.loc["SARAL_pass1_detected", "track_found"]
    assert s.loc["SARAL_pass1_detected", "signal"] >= 30          # the in-sky snapshots
    assert s.loc["SARAL_pass1_detected", "noise"] == 20           # notinsky + below horizon
    assert s.loc["LILACSAT-2_not_detected", "signal"] == 0 and s.loc["LILACSAT-2_not_detected", "noise"] == 24
    iq_train = pd.read_csv(tmp_path / "out" / "client_passes_iq_train.csv")
    iq_ext = pd.read_csv(tmp_path / "out" / "client_passes_iq_external.csv")
    assert set(iq_ext["pass"]) == {"SARAL_pass2_detected"}
    assert set(iq_ext.label) == {0, 1}          # the rising part before the track gives negatives
    assert "SARAL_pass2_detected" not in set(iq_train["pass"])
    assert set(b.FEATURE_NAMES) <= set(iq_train.columns) and iq_train.recording_id.notna().all()
    # labels agree with where the tone really was
    sig_rows = iq_train[iq_train.label == 1]
    assert not sig_rows.notes.str.contains("below").any()
    wf = pd.read_csv(tmp_path / "out" / "client_passes_waterfall_train.csv")
    assert set(b.WATERFALL_FEATURE_NAMES) <= set(wf.columns) and set(wf.label) <= {0, 1}
    assert (tmp_path / "work" / "preview" / "SARAL_pass1_detected.png").exists()
    # second run reuses the cache and gives the same rows
    r2 = b.build(aa, tmp_path / "out2", tmp_path / "work", snapshot_s=0.2)
    assert pd.read_csv(tmp_path / "out2" / "client_passes_iq_train.csv").equals(iq_train)


def test_segment_states():
    assert b.segment_state("notinsky") == "below"
    assert b.segment_state("notinskyto3") == "mixed"
    assert b.segment_state("-5to-10") == "below"
    assert b.segment_state("5to-5") == "mixed"
    for name in ("justrisen", "max35", "maxto17", "15toset", "8tomax", "21to62"):
        assert b.segment_state(name) == "above"


def test_spur_and_dc_are_notched():
    x = np.zeros(NB)
    x[700] = 9
    spurs = b.find_spurs(x)
    assert spurs[700] and spurs[NB // 2] and not spurs[300]


def test_original_client_folder_names_still_work():
    assert b.pass_kind("Nondetection(LILACSAT2)") == "nondetection" and b.pass_satellite("Nondetection(LILACSAT2)") == "LILACSAT2"
    assert b.pass_kind("MarginalDetection(NORAD-16)") == "marginal"
    assert b.pass_kind("DetectedSatellite2(SARAL)") == "detected"
    assert b.pass_kind("LILACSAT-2_not_detected") == "nondetection" and b.pass_satellite("LILACSAT-2_not_detected") == "LILACSAT-2"
    assert b.pass_kind("SARAL_pass2_detected") == "detected" and b.pass_satellite("SARAL_pass2_detected") == "SARAL"


def test_a_fixed_line_is_not_a_satellite_track(tmp_path):
    """NORAD-16 case: interference that appears mid-pass at one frequency must not become 'signal'."""
    rng = np.random.default_rng(3)
    root = tmp_path / "recs"
    d = root / "NORAD-16_marginal"
    k = 0
    for folder, n, line in (("notinsky", 15, False), ("12to41", 25, True), ("40to15", 25, True)):
        (d / folder).mkdir(parents=True)
        for i in range(n):
            m = _snapshot(rng)
            if line:
                m[:, NB // 2 + 9] += 12          # steady +2 kHz line, absent before rise
            f = d / folder / f"spectrogram{i}_465988000.txt"
            np.savetxt(f, m, fmt="%.2f")
            os.utime(f, (1e6 + 1.6 * k, 1e6 + 1.6 * k))
            k += 1
    r = b.build(root, tmp_path / "out", tmp_path / "work", external=[], snapshot_s=0.2)
    s = r["summary"].iloc[0]
    assert not s.track_found and s.signal == 0 and s.noise == 15


def test_persistent_spurs():
    spectra = np.zeros((100, NB))
    spectra[:, 600] += 9                       # always there -> spur
    spectra[np.arange(100), 400 + np.arange(100)] += 9   # moving trace -> not a spur
    sp = b.persistent_spurs(spectra)
    assert sp[600] and not sp[450]
