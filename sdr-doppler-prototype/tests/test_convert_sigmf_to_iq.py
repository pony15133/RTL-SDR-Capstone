"""Tests for scripts/convert_sigmf_to_iq.py.

scripts/ isn't a package, so the module is loaded by file path. All
.sigmf-meta/.sigmf-data fixtures here are hand-constructed minimal
examples built purely to exercise the datatype parsing/scaling/
deinterleaving logic - not real SigMF captures.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "convert_sigmf_to_iq.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("convert_sigmf_to_iq", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


csi = _load_script_module()


def _write_meta(path: Path, datatype: str, sample_rate=None, frequency=None) -> None:
    meta = {"global": {"core:datatype": datatype}}
    if sample_rate is not None:
        meta["global"]["core:sample_rate"] = sample_rate
    if frequency is not None:
        meta["captures"] = [{"core:frequency": frequency, "core:sample_start": 0}]
    path.write_text(json.dumps(meta), encoding="utf-8")


class TestParseSigmfDatatype:
    def test_ci16_le(self):
        is_complex, dtype = csi.parse_sigmf_datatype("ci16_le")
        assert is_complex is True
        assert dtype == np.dtype("<i2")

    def test_cu8(self):
        is_complex, dtype = csi.parse_sigmf_datatype("cu8")
        assert is_complex is True
        assert dtype == np.dtype("u1")

    def test_cf32_le(self):
        is_complex, dtype = csi.parse_sigmf_datatype("cf32_le")
        assert is_complex is True
        assert dtype == np.dtype("<f4")

    def test_real_valued_datatype_detected(self):
        is_complex, dtype = csi.parse_sigmf_datatype("ri16_le")
        assert is_complex is False

    def test_big_endian(self):
        is_complex, dtype = csi.parse_sigmf_datatype("ci16_be")
        assert dtype == np.dtype(">i2")

    def test_unrecognised_string_raises(self):
        with pytest.raises(csi.SigMFFormatError):
            csi.parse_sigmf_datatype("not_a_real_datatype")


class TestConvertSigmfCi16:
    def test_known_values_scale_correctly(self, tmp_path):
        meta_path = tmp_path / "cap.sigmf-meta"
        data_path = tmp_path / "cap.sigmf-data"
        _write_meta(meta_path, "ci16_le", sample_rate=4_000_000, frequency=401_300_000)

        i_vals = np.array([0, 16384, -32768, 32767], dtype="<i2")
        q_vals = np.array([0, -16384, 32767, -32768], dtype="<i2")
        interleaved = np.empty(8, dtype="<i2")
        interleaved[0::2] = i_vals
        interleaved[1::2] = q_vals
        interleaved.tofile(data_path)

        output_path = tmp_path / "out.npy"
        info = csi.convert_sigmf(meta_path, data_path, output_path)

        iq = np.load(output_path)
        assert iq.dtype == np.complex64
        assert iq.shape == (4,)
        np.testing.assert_allclose(iq.real, [0.0, 0.5, -1.0, 32767 / 32768], atol=1e-6)
        np.testing.assert_allclose(iq.imag, [0.0, -0.5, 32767 / 32768, -1.0], atol=1e-6)

        assert info["datatype"] == "ci16_le"
        assert info["n_samples"] == 4
        assert info["sample_rate_hz"] == 4_000_000
        assert info["center_freq_hz"] == 401_300_000


class TestConvertSigmfCu8:
    def test_rtl_sdr_style_centring(self, tmp_path):
        """cu8 is RTL-SDR's own native format - DC-centred at 127.5."""
        meta_path = tmp_path / "cap.sigmf-meta"
        data_path = tmp_path / "cap.sigmf-data"
        _write_meta(meta_path, "cu8")

        # I=0 -> most negative, I=255 -> most positive, I=127/128 -> near zero
        raw = np.array([0, 127, 128, 255, 127, 255, 0, 128], dtype="u1")
        raw.tofile(data_path)

        output_path = tmp_path / "out.npy"
        csi.convert_sigmf(meta_path, data_path, output_path)
        iq = np.load(output_path)

        midpoint = 127.5
        expected_i = (np.array([0, 128, 127, 0], dtype=np.float32) - midpoint) / midpoint
        expected_q = (np.array([127, 255, 255, 128], dtype=np.float32) - midpoint) / midpoint
        np.testing.assert_allclose(iq.real, expected_i, atol=1e-6)
        np.testing.assert_allclose(iq.imag, expected_q, atol=1e-6)
        assert np.all(np.abs(iq.real) <= 1.0)
        assert np.all(np.abs(iq.imag) <= 1.0)


class TestConvertSigmfCf32:
    def test_float_passthrough_unscaled(self, tmp_path):
        meta_path = tmp_path / "cap.sigmf-meta"
        data_path = tmp_path / "cap.sigmf-data"
        _write_meta(meta_path, "cf32_le")

        interleaved = np.array([0.1, 0.2, -0.5, 0.9], dtype="<f4")
        interleaved.tofile(data_path)

        output_path = tmp_path / "out.npy"
        csi.convert_sigmf(meta_path, data_path, output_path)
        iq = np.load(output_path)

        np.testing.assert_allclose(iq.real, [0.1, -0.5], atol=1e-6)
        np.testing.assert_allclose(iq.imag, [0.2, 0.9], atol=1e-6)


class TestConvertSigmfErrors:
    def test_missing_datatype_field_raises(self, tmp_path):
        meta_path = tmp_path / "cap.sigmf-meta"
        data_path = tmp_path / "cap.sigmf-data"
        meta_path.write_text(json.dumps({"global": {}}))
        np.array([0, 0], dtype="<i2").tofile(data_path)

        with pytest.raises(csi.SigMFFormatError, match="core:datatype"):
            csi.convert_sigmf(meta_path, data_path, tmp_path / "out.npy")

    def test_real_valued_datatype_rejected(self, tmp_path):
        meta_path = tmp_path / "cap.sigmf-meta"
        data_path = tmp_path / "cap.sigmf-data"
        _write_meta(meta_path, "ri16_le")
        np.array([0, 1, 2, 3], dtype="<i2").tofile(data_path)

        with pytest.raises(csi.SigMFFormatError, match="real-valued"):
            csi.convert_sigmf(meta_path, data_path, tmp_path / "out.npy")

    def test_odd_sample_count_raises(self, tmp_path):
        meta_path = tmp_path / "cap.sigmf-meta"
        data_path = tmp_path / "cap.sigmf-data"
        _write_meta(meta_path, "ci16_le")
        np.array([0, 1, 2], dtype="<i2").tofile(data_path)  # odd count - can't be I/Q pairs

        with pytest.raises(csi.SigMFFormatError, match="odd"):
            csi.convert_sigmf(meta_path, data_path, tmp_path / "out.npy")

    def test_empty_data_file_raises(self, tmp_path):
        meta_path = tmp_path / "cap.sigmf-meta"
        data_path = tmp_path / "cap.sigmf-data"
        _write_meta(meta_path, "ci16_le")
        data_path.touch()

        with pytest.raises(csi.SigMFFormatError, match="empty"):
            csi.convert_sigmf(meta_path, data_path, tmp_path / "out.npy")

    def test_unrecognised_datatype_string_raises(self, tmp_path):
        meta_path = tmp_path / "cap.sigmf-meta"
        data_path = tmp_path / "cap.sigmf-data"
        _write_meta(meta_path, "totally_bogus")
        np.array([0, 1], dtype="<i2").tofile(data_path)

        with pytest.raises(csi.SigMFFormatError):
            csi.convert_sigmf(meta_path, data_path, tmp_path / "out.npy")

    def test_invalid_json_meta_raises(self, tmp_path):
        meta_path = tmp_path / "cap.sigmf-meta"
        data_path = tmp_path / "cap.sigmf-data"
        meta_path.write_text("{not valid json")
        np.array([0, 1], dtype="<i2").tofile(data_path)

        with pytest.raises(csi.SigMFFormatError):
            csi.convert_sigmf(meta_path, data_path, tmp_path / "out.npy")


class TestCliMain:
    def test_missing_meta_file_fails_cleanly(self, tmp_path):
        exit_code = csi.main([
            "--meta", str(tmp_path / "missing.sigmf-meta"),
            "--data", str(tmp_path / "missing.sigmf-data"),
            "--output", str(tmp_path / "out.npy"),
        ])
        assert exit_code == 1

    def test_successful_conversion_end_to_end(self, tmp_path):
        meta_path = tmp_path / "cap.sigmf-meta"
        data_path = tmp_path / "cap.sigmf-data"
        _write_meta(meta_path, "ci16_le", sample_rate=2_400_000)
        np.array([100, 200, -100, -200], dtype="<i2").tofile(data_path)
        output_path = tmp_path / "out.npy"

        exit_code = csi.main(["--meta", str(meta_path), "--data", str(data_path), "--output", str(output_path)])

        assert exit_code == 0
        assert output_path.exists()
        iq = np.load(output_path)
        assert iq.dtype == np.complex64
        assert iq.shape == (2,)
