import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable


class MaterializeRvdiOnlyTests(unittest.TestCase):
    def test_recomputes_only_rvdi_margin_and_registers_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            row = {
                "record_id": "one", "prompt": "What?", "chosen": "yes", "rejected": "no",
                "images": ["/tmp/example.jpg"], "percentile": 0.5, "rvdi_lcb": 0.1,
                "vdi": 0.1, "delta_h": 2.0, "lambda_i": 0.19, "margin": 0.38,
                "relation_confidence": 0.8, "rvdi_std_normalized": 0.9, "length_risk": 0.7,
                "capabilities": ["general"],
            }
            train = base / "train.json"
            dev = base / "dev.json"
            train.write_text(json.dumps([row]))
            dev.write_text(json.dumps([{**row, "record_id": "two"}]))
            out = base / "output"
            result = subprocess.run(
                [PYTHON, str(ROOT / "scripts/car_predpo/materialize_rvdi_only.py"),
                 "--train", str(train), "--dev", str(dev), "--output-dir", str(out),
                 "--expected-train", "1", "--expected-dev", "1"],
                cwd=ROOT, text=True, capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            actual = json.loads((out / "car_predpo_train_rvdi_only.json").read_text())[0]
            self.assertAlmostEqual(actual["lambda_i"], 0.115)
            self.assertAlmostEqual(actual["margin"], 0.23)
            self.assertEqual(actual["lambda_i_original"], 0.19)
            self.assertEqual(actual["relation_factor"], 1.0)
            self.assertEqual(actual["uncertainty_factor"], 1.0)
            self.assertEqual(actual["length_factor"], 1.0)
            info = json.loads((out / "dataset_info.json").read_text())
            self.assertEqual(set(info["car_predpo_train_rvdi_only"]["columns"]),
                             {"prompt", "chosen", "rejected", "images", "lambda_i", "delta_h", "vdi"})
            self.assertEqual(len(json.loads((out / "car_predpo_dev_rvdi_only.json").read_text())), 1)


if __name__ == "__main__":
    unittest.main()
