import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "car_predpo" / "materialize_shuffled_lambda_car.py"


class ShuffledLambdaMaterializationTests(unittest.TestCase):
    def test_shuffles_only_train_lambdas_and_recomputes_train_margins(self):
        """Catches implementations that shuffle dev data, alter the lambda multiset, or retain stale margins."""
        train = [
            {"record_id": "train-1", "lambda_i": 0.10, "delta_h": 1.0, "margin": 0.10},
            {"record_id": "train-2", "lambda_i": 0.20, "delta_h": 2.0, "margin": 0.40},
            {"record_id": "train-3", "lambda_i": 0.30, "delta_h": 0.5, "margin": 0.15},
            {"record_id": "train-4", "lambda_i": 0.40, "delta_h": 3.0, "margin": 1.20},
        ]
        dev = [
            {"record_id": "dev-1", "lambda_i": 0.25, "delta_h": 4.0, "margin": 1.00},
            {"record_id": "dev-2", "lambda_i": 0.35, "delta_h": 2.0, "margin": 0.70},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_path = root / "train.json"
            dev_path = root / "dev.json"
            first_output = root / "first"
            second_output = root / "second"
            train_path.write_text(json.dumps(train), encoding="utf-8")
            dev_path.write_text(json.dumps(dev), encoding="utf-8")
            command = [
                sys.executable,
                str(SCRIPT),
                "--train",
                str(train_path),
                "--dev",
                str(dev_path),
                "--expected-train",
                "4",
                "--expected-dev",
                "2",
                "--permutation-seed",
                "42",
            ]
            first = subprocess.run(
                [*command, "--output-dir", str(first_output)],
                cwd=ROOT,
                text=True,
                capture_output=True,
            )
            self.assertEqual(first.returncode, 0, first.stderr)
            second = subprocess.run(
                [*command, "--output-dir", str(second_output)],
                cwd=ROOT,
                text=True,
                capture_output=True,
            )
            self.assertEqual(second.returncode, 0, second.stderr)

            shuffled_train = json.loads((first_output / "car_predpo_train_shuffled_lambda.json").read_text())
            shuffled_dev = json.loads((first_output / "car_predpo_dev_shuffled_lambda.json").read_text())
            repeated_train = json.loads((second_output / "car_predpo_train_shuffled_lambda.json").read_text())
            manifest = json.loads((first_output / "shuffled_lambda_manifest.json").read_text())

        self.assertEqual(shuffled_dev, dev)
        self.assertEqual(sorted(row["lambda_i"] for row in shuffled_train), [0.10, 0.20, 0.30, 0.40])
        self.assertNotEqual([row["lambda_i"] for row in shuffled_train], [row["lambda_i"] for row in train])
        self.assertEqual([row["lambda_i"] for row in repeated_train], [row["lambda_i"] for row in shuffled_train])
        for source, shuffled in zip(train, shuffled_train):
            self.assertEqual(shuffled["record_id"], source["record_id"])
            self.assertEqual(shuffled["delta_h"], source["delta_h"])
            self.assertEqual(shuffled["lambda_i_original"], source["lambda_i"])
            self.assertEqual(shuffled["margin_original"], source["margin"])
            self.assertAlmostEqual(shuffled["margin"], shuffled["lambda_i"] * source["delta_h"])
        self.assertEqual(manifest["permutation_seed"], 42)
        self.assertEqual(manifest["counts"], {"train": 4, "dev": 2})


if __name__ == "__main__":
    unittest.main()
