import json
import tempfile
import unittest
from pathlib import Path

from car_predpo.dataset import materialize_scored_split


class ScoredSplitMaterializationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.scored = self.root / "scored.json"
        self.split = self.root / "split_index.json"
        self.output = self.root / "output"

    def tearDown(self):
        self.tmp.cleanup()

    def write_inputs(self, rows, split):
        self.scored.write_text(json.dumps(rows), encoding="utf-8")
        self.split.write_text(json.dumps(split), encoding="utf-8")

    def test_materializes_disjoint_train_and_dev_with_manifest(self):
        rows = [
            {"record_id": "r1", "image_key": "i1", "prompt": "p1"},
            {"record_id": "r2", "image_key": "i2", "prompt": "p2"},
            {"record_id": "r3", "image_key": "i3", "prompt": "p3"},
        ]
        self.write_inputs(rows, {"r1": "train", "r2": "train", "r3": "dev"})

        report = materialize_scored_split(
            self.scored, self.split, self.output, expected_train=2, expected_dev=1
        )

        train = json.loads((self.output / "car_predpo_train_scored.json").read_text())
        dev = json.loads((self.output / "car_predpo_dev_scored.json").read_text())
        dataset_info = json.loads((self.output / "dataset_info.json").read_text())
        self.assertEqual([row["record_id"] for row in train], ["r1", "r2"])
        self.assertEqual([row["record_id"] for row in dev], ["r3"])
        self.assertEqual(report["train_count"], 2)
        self.assertEqual(report["dev_count"], 1)
        self.assertIn("car_predpo_train_scored", dataset_info)
        self.assertIn("car_predpo_dev_scored", dataset_info)

    def test_rejects_unknown_split_id(self):
        self.write_inputs(
            [{"record_id": "r1", "image_key": "i1"}],
            {"r1": "train", "missing": "dev"},
        )
        with self.assertRaisesRegex(ValueError, "unknown record_id"):
            materialize_scored_split(self.scored, self.split, self.output)

    def test_rejects_duplicate_record_id(self):
        self.write_inputs(
            [
                {"record_id": "r1", "image_key": "i1"},
                {"record_id": "r1", "image_key": "i2"},
            ],
            {"r1": "train"},
        )
        with self.assertRaisesRegex(ValueError, "duplicate record_id"):
            materialize_scored_split(self.scored, self.split, self.output)

    def test_rejects_image_overlap(self):
        self.write_inputs(
            [
                {"record_id": "r1", "image_key": "shared"},
                {"record_id": "r2", "image_key": "shared"},
            ],
            {"r1": "train", "r2": "dev"},
        )
        with self.assertRaisesRegex(ValueError, "image overlap"):
            materialize_scored_split(self.scored, self.split, self.output)


if __name__ == "__main__":
    unittest.main()
