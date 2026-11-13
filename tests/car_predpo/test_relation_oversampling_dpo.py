import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from car_predpo.dataset import materialize_relation_oversampling_pair


class RelationOversamplingMaterializationTests(unittest.TestCase):
    def make_source_dataset(self, root: Path) -> Path:
        source = root / "source"
        source.mkdir()
        train = [
            {"record_id": "car-000001", "image_key": "image-1", "relation_confidence": 0.0, "lambda_i": 0.1, "lambda_raw": 0.1, "margin": 0.2, "delta_h": 2.0, "vdi": 0.3},
            {"record_id": "car-000002", "image_key": "image-2", "relation_confidence": 1.0, "lambda_i": 0.2, "lambda_raw": 0.2, "margin": 0.4, "delta_h": 2.0, "vdi": 0.4},
            {"record_id": "car-000003", "image_key": "image-3", "relation_confidence": 0.0, "lambda_i": 0.3, "lambda_raw": 0.3, "margin": 0.6, "delta_h": 2.0, "vdi": 0.5},
        ]
        dev = [{"record_id": "car-000004", "image_key": "image-4", "relation_confidence": 1.0, "lambda_i": 0.4, "lambda_raw": 0.4, "margin": 0.8, "delta_h": 2.0, "vdi": 0.6}]
        (source / "train.json").write_text(json.dumps(train), encoding="utf-8")
        (source / "dev.json").write_text(json.dumps(dev), encoding="utf-8")
        (source / "dataset_info.json").write_text(json.dumps({
            "source_train": {"file_name": "train.json", "columns": {"lambda_i": "lambda_i", "delta_h": "delta_h", "vdi": "vdi"}},
            "source_dev": {"file_name": "dev.json", "columns": {"lambda_i": "lambda_i", "delta_h": "delta_h", "vdi": "vdi"}},
        }), encoding="utf-8")
        return source

    def test_expands_relation_rows_and_zeros_only_dpo_margin_fields(self):
        """Would fail if relations are not duplicated or DPO+RO receives a nonzero margin."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.make_source_dataset(root)
            output = root / "paired"
            report = materialize_relation_oversampling_pair(source, output)
            sequence = json.loads((output / "sampled_index_sequence.json").read_text(encoding="utf-8"))
            full = json.loads((output / "car_predpo_train_relation_oversampled_full.json").read_text(encoding="utf-8"))
            dpo = json.loads((output / "car_predpo_train_relation_oversampled_dpo.json").read_text(encoding="utf-8"))

        self.assertEqual(sequence, ["car-000001", "car-000002", "car-000002", "car-000003"])
        self.assertEqual([row["record_id"] for row in full], sequence)
        self.assertEqual([row["record_id"] for row in dpo], sequence)
        self.assertEqual([row["lambda_i"] for row in dpo], [0.0, 0.0, 0.0, 0.0])
        self.assertEqual([row["lambda_raw"] for row in dpo], [0.0, 0.0, 0.0, 0.0])
        self.assertEqual([row["margin"] for row in dpo], [0.0, 0.0, 0.0, 0.0])
        self.assertEqual(full[1]["lambda_i"], 0.2)
        self.assertEqual(dpo[1]["lambda_i_original"], 0.2)
        self.assertEqual(report["relation_train_count"], 1)
        self.assertEqual(report["sequence_count"], 4)

    def test_rejects_non_binary_relation_confidence(self):
        """Would fail if ambiguous relation labels silently alter sampling probability."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.make_source_dataset(root)
            train_path = source / "train.json"
            train = json.loads(train_path.read_text(encoding="utf-8"))
            train[1]["relation_confidence"] = 0.5
            train_path.write_text(json.dumps(train), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "relation_confidence must be 0.0 or 1.0"):
                materialize_relation_oversampling_pair(source, root / "paired")


    def test_preflight_rejects_nonzero_dpo_margin(self):
        """Would fail if a DPO+RO row could silently regain an adaptive margin."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.make_source_dataset(root)
            paired = root / "paired"
            materialize_relation_oversampling_pair(source, paired)
            dpo_path = paired / "car_predpo_train_relation_oversampled_dpo.json"
            dpo = json.loads(dpo_path.read_text(encoding="utf-8"))
            dpo[1]["margin"] = 0.1
            dpo_path.write_text(json.dumps(dpo), encoding="utf-8")
            env = os.environ.copy()
            env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
            result = subprocess.run(
                [sys.executable, str(Path(__file__).resolve().parents[2] / "scripts/car_predpo/validate_relation_oversampling_pair.py"),
                 "--source-dataset-dir", str(source), "--paired-dataset-dir", str(paired)],
                text=True, capture_output=True, env=env,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("DPO+RO margin field must be zero", result.stderr)


    def test_preflight_rejects_tampered_dataset_registration(self):
        """Would fail if Stage 2 could consume files other than those the pair validator checked."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.make_source_dataset(root)
            paired = root / "paired"
            materialize_relation_oversampling_pair(source, paired)
            info_path = paired / "dataset_info.json"
            info = json.loads(info_path.read_text(encoding="utf-8"))
            info["car_predpo_train_relation_oversampled_dpo"]["file_name"] = "car_predpo_train_relation_oversampled_full.json"
            info_path.write_text(json.dumps(info), encoding="utf-8")
            env = os.environ.copy()
            env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
            result = subprocess.run(
                [sys.executable, str(Path(__file__).resolve().parents[2] / "scripts/car_predpo/validate_relation_oversampling_pair.py"),
                 "--source-dataset-dir", str(source), "--paired-dataset-dir", str(paired)],
                text=True, capture_output=True, env=env,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("paired dataset registration mismatch", result.stderr)


    def test_rejects_source_rows_with_reserved_margin_audit_fields(self):
        """Would fail if DPO+RO overwrote existing source provenance during materialization."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.make_source_dataset(root)
            train_path = source / "train.json"
            train = json.loads(train_path.read_text(encoding="utf-8"))
            train[0]["lambda_i_original"] = 0.9
            train_path.write_text(json.dumps(train), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "reserved DPO\\+RO audit field"):
                materialize_relation_oversampling_pair(source, root / "paired")


if __name__ == "__main__":
    unittest.main()
