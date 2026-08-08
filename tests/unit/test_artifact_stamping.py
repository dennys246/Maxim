"""Regression guard for artifact stamping (1.1 item 7, fabric plan Stage-4 pull-forward).

THE LEAK THIS PINS: `LinguisticEncoder` at the same configured model name emits
768-dim real vectors OR 384-dim bag-of-words hashes depending on whether the
`semantic` extra is installed, and `_sensor_embed` behaves differently under
range-aware vs range-blind normalization — none of which was recorded anywhere.
A substrate bundle composed from a fallback-encoded session circulated as if
comparable to a real-encoder one; the truth was unrecoverable post-hoc (a
384-dim array could be either). Stamps are recorded at ENCODE time by the code
that knows (`EC.record_encoder_provenance`), persisted through `EC.save/load`,
and carried into bundle manifests alongside dims DERIVED from the actual arrays.
"""

from __future__ import annotations

import json
import zipfile

import pytest

from maxim.similarity.ec import EntorhinalCortex


def _make_ec() -> EntorhinalCortex:
    return EntorhinalCortex()


class TestECProvenanceRecording:
    def test_record_and_read_back(self):
        ec = _make_ec()
        ec.record_encoder_provenance("linguistic", {"model_name": "m", "using_fallback": True, "embedding_dim": 384})
        prov = ec.encoder_provenance
        assert prov["linguistic"]["using_fallback"] is True
        assert prov["linguistic"]["embedding_dim"] == 384

    def test_sensor_names_accumulate_as_union(self):
        ec = _make_ec()
        ec.record_encoder_provenance("sensor:interoception", {"sensor_names": ["cold", "hunger"]})
        ec.record_encoder_provenance("sensor:interoception", {"sensor_names": ["cold", "azimuth"]})
        assert ec.encoder_provenance["sensor:interoception"]["sensor_names"] == ["azimuth", "cold", "hunger"]

    def test_mixed_normalization_modes_are_visible(self):
        """A session that mixed range-aware and range-blind calls must SAY so —
        collapsing to last-write would hide exactly the calibration difference
        the stamp exists to expose."""
        ec = _make_ec()
        ec.record_encoder_provenance("sensor:audio", {"normalization": "range-aware"})
        ec.record_encoder_provenance("sensor:audio", {"normalization": "range-blind"})
        assert ec.encoder_provenance["sensor:audio"]["normalization_modes"] == ["range-aware", "range-blind"]

    def test_survives_save_load_round_trip(self, tmp_path):
        ec = _make_ec()
        ec.record_encoder_provenance("linguistic", {"using_fallback": False, "embedding_dim": 768})
        path = str(tmp_path / "ec.json")
        ec.save(path)
        ec2 = _make_ec()
        ec2.load(path)
        assert ec2.encoder_provenance["linguistic"]["embedding_dim"] == 768

    def test_pre_stamping_file_loads_with_empty_provenance(self, tmp_path):
        ec = _make_ec()
        path = str(tmp_path / "ec.json")
        ec.save(path)
        data = json.loads((tmp_path / "ec.json").read_text())
        data.pop("encoder_provenance", None)
        (tmp_path / "ec.json").write_text(json.dumps(data))
        ec2 = _make_ec()
        ec2.load(path)
        assert ec2.encoder_provenance == {}


class TestEncoderRecordingSites:
    def test_linguistic_embed_records_realized_state(self):
        from maxim.similarity.encoder import LinguisticEncoder

        ec = _make_ec()
        enc = LinguisticEncoder(ec=ec, atl=None)
        vec = enc.embed("a small test phrase")
        prov = ec.encoder_provenance["linguistic"]
        assert prov["embedding_dim"] == len(vec), "dim must be MEASURED on the actual vector, not declared"
        assert prov["using_fallback"] == enc.using_fallback

    def test_sensor_encoder_records_names_and_mode(self):
        from maxim.similarity.encoder import SensorEncoder

        ec = _make_ec()
        enc = SensorEncoder(ec=ec)
        enc.encode_sensors(
            agent_id="a",
            sensors={"azimuth": 0.5, "cold": 0.2},
            modality="audio",
            ranges={"azimuth": (-1.0, 1.0), "cold": (0.0, 1.0)},
        )
        prov = ec.encoder_provenance["sensor:audio"]
        assert prov["sensor_names"] == ["azimuth", "cold"]
        assert prov["normalization_modes"] == ["range-aware"]

    def test_sensor_encoder_partial_ranges_stamp_range_partial(self):
        from maxim.similarity.encoder import SensorEncoder

        ec = _make_ec()
        enc = SensorEncoder(ec=ec)
        enc.encode_sensors(
            agent_id="a", sensors={"azimuth": 0.5, "cold": 0.2}, modality="audio", ranges={"azimuth": (-1.0, 1.0)}
        )
        assert ec.encoder_provenance["sensor:audio"]["normalization_modes"] == ["range-partial"]

    def test_sensor_encoder_no_ranges_stamps_range_blind(self):
        from maxim.similarity.encoder import SensorEncoder

        ec = _make_ec()
        enc = SensorEncoder(ec=ec)
        enc.encode_sensors(agent_id="a", sensors={"cold": 0.2}, modality="interoception")
        assert ec.encoder_provenance["sensor:interoception"]["normalization_modes"] == ["range-blind"]


class TestBundleStamping:
    def _nodes(self):
        return {
            "n1": {"embedding": [0.1] * 384, "modality": "linguistic", "count": 1, "source": "local", "domain": None},
            "n2": {"embedding": [0.2] * 768, "modality": "linguistic", "count": 1, "source": "local", "domain": None},
            "n3": {"embedding": [0.3] * 64, "modality": "interoception", "count": 1, "source": "local", "domain": None},
        }

    def test_observed_dims_derived_from_actual_arrays(self, tmp_path):
        from maxim.hivemind.bundle import compose_bundle

        manifest = compose_bundle(
            nac_state=None,
            ec_substrate_nodes=self._nodes(),
            output_path=tmp_path / "b.zip",
            contributor_id="tester",
        )
        dims = manifest["encoder_provenance"]["observed_embedding_dims"]
        assert dims["linguistic"] == [384, 768], (
            "a mixed-dim slice (the #467 corruption class) must be VISIBLE in the manifest"
        )
        assert dims["interoception"] == [64]

    def test_recorded_provenance_passes_through_and_none_is_honest(self, tmp_path):
        from maxim.hivemind.bundle import compose_bundle

        recorded = {"linguistic": {"using_fallback": True, "embedding_dim": 384}}
        manifest = compose_bundle(
            nac_state=None,
            ec_substrate_nodes=self._nodes(),
            output_path=tmp_path / "b1.zip",
            contributor_id="tester",
            encoder_provenance=recorded,
        )
        assert manifest["encoder_provenance"]["recorded"] == recorded

        manifest2 = compose_bundle(
            nac_state=None,
            ec_substrate_nodes=self._nodes(),
            output_path=tmp_path / "b2.zip",
            contributor_id="tester",
        )
        assert manifest2["encoder_provenance"]["recorded"] is None, (
            "a pre-stamping payload must carry an honest unknown, never a fabricated default"
        )

    def test_stamp_lands_in_the_zip_manifest(self, tmp_path):
        from maxim.hivemind.bundle import compose_bundle

        out = tmp_path / "b.zip"
        compose_bundle(nac_state=None, ec_substrate_nodes=self._nodes(), output_path=out, contributor_id="tester")
        with zipfile.ZipFile(out) as zf:
            manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
        assert "encoder_provenance" in manifest

    def test_nac_only_bundle_has_empty_observed_dims(self, tmp_path):
        from maxim.hivemind.bundle import compose_bundle

        manifest = compose_bundle(
            nac_state={"links": {}},
            ec_substrate_nodes=None,
            output_path=tmp_path / "b.zip",
            contributor_id="tester",
        )
        assert manifest["encoder_provenance"]["observed_embedding_dims"] == {}


class TestExtractCompat:
    def _nodes(self):
        return {
            "n1": {"embedding": [0.1] * 384, "modality": "linguistic", "count": 1, "source": "local", "domain": None},
        }

    def test_extract_bundle_accepts_stamped_manifest(self, tmp_path):
        """The stamped-EC path, actually exercised (review fold: the first
        version's hasattr hack silently degraded this to a nac-only bundle
        with no EC slice — an instrument that could not detect the
        regression it appeared to cover)."""
        from maxim.hivemind.bundle import compose_bundle, extract_bundle

        out = tmp_path / "b.zip"
        compose_bundle(
            nac_state={"links": {}},
            ec_substrate_nodes=self._nodes(),
            output_path=out,
            contributor_id="tester",
            encoder_provenance={"linguistic": {"using_fallback": False}},
        )
        manifest = extract_bundle(bundle_path=out, output_dir=tmp_path / "extracted")
        assert manifest is not None
        assert manifest["encoder_provenance"]["observed_embedding_dims"] == {"linguistic": [384]}


class TestCliExportPassthrough:
    """The read-from-payload-not-live-singleton seam (review fold F5): the
    CLI must carry the WRITER's stamps from aut_ec.json — falling back to
    the reader's own encoder singleton would stamp the reader's calibration
    onto the writer's arrays, the precise leak the stamp exists to prevent."""

    def _run_export(self, session_dir, out_path):
        import argparse

        from maxim.hivemind.cli import _run_export

        args = argparse.Namespace(
            session=str(session_dir),
            output=str(out_path),
            contributor_id="tester",
            domain=None,
            no_identity_filter=False,
            identity_threshold=2,
        )
        return _run_export(args)

    def test_export_carries_writer_stamps(self, tmp_path):
        session = tmp_path / "session"
        session.mkdir()
        recorded = {"linguistic": {"using_fallback": True, "embedding_dim": 384}}
        (session / "aut_ec.json").write_text(
            json.dumps(
                {
                    "substrate_nodes": {
                        "n1": {"embedding": [0.1] * 384, "modality": "linguistic", "count": 1},
                    },
                    "encoder_provenance": recorded,
                }
            )
        )
        out = tmp_path / "b.zip"
        assert self._run_export(session, out) == 0
        with zipfile.ZipFile(out) as zf:
            manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
        assert manifest["encoder_provenance"]["recorded"] == recorded

    def test_export_of_pre_stamping_session_carries_none(self, tmp_path):
        session = tmp_path / "session"
        session.mkdir()
        (session / "aut_ec.json").write_text(
            json.dumps(
                {
                    "substrate_nodes": {
                        "n1": {"embedding": [0.1] * 384, "modality": "linguistic", "count": 1},
                    }
                }
            )
        )
        out = tmp_path / "b.zip"
        assert self._run_export(session, out) == 0
        with zipfile.ZipFile(out) as zf:
            manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
        assert manifest["encoder_provenance"]["recorded"] is None, (
            "pre-stamping session must export an honest unknown — NEVER the reader's own encoder state"
        )
