import csv
import json
import tempfile
import unittest
from pathlib import Path

from car_predpo.rvdi_human_validation import analyze_annotation_package, prepare_annotation_package


class RVDIHumanValidationPreparationTests(unittest.TestCase):
    def test_preparation_stratifies_hides_scores_and_is_seed_reproducible(self):
        """Catches unstratified sampling, RVDI leakage into blind forms, and non-reproducible reviewer packets."""
        rows = [
            {
                "record_id": f"r{index}",
                "rvdi_lcb": float(index),
                "images": [f"/images/{index}.jpg"],
                "prompt": f"question {index}",
                "chosen": f"answer {index}",
            }
            for index in range(10)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "dev_scored.json"
            first = root / "first"
            second = root / "second"
            source.write_text(json.dumps(rows), encoding="utf-8")

            prepare_annotation_package(source, first, sample_size=5, seed=42, strata=5)
            prepare_annotation_package(source, second, sample_size=5, seed=42, strata=5)

            with (first / "control" / "annotation_key.csv").open(encoding="utf-8", newline="") as handle:
                key_rows = list(csv.DictReader(handle))
            with (first / "reviewer" / "annotator_a.csv").open(encoding="utf-8", newline="") as handle:
                blind_rows = list(csv.DictReader(handle))

            self.assertEqual(len(key_rows), 5)
            self.assertEqual(sorted(row["rvdi_quintile"] for row in key_rows), ["1", "2", "3", "4", "5"])
            self.assertEqual(
                (first / "control" / "annotation_key.csv").read_text(),
                (second / "control" / "annotation_key.csv").read_text(),
            )
            self.assertEqual([row["record_id"] for row in blind_rows], [row["record_id"] for row in key_rows])
            self.assertNotIn("rvdi_lcb", blind_rows[0])
            self.assertNotIn("rvdi_quintile", blind_rows[0])
            self.assertEqual(blind_rows[0]["visual_dependence"], "")
            self.assertFalse((first / "reviewer" / "annotation_key.csv").exists())


    def test_analysis_reports_perfect_agreement_and_positive_rank_correlation(self):
        """Catches analyses that ignore reviewer agreement, misalign IDs, or omit the RVDI-validity statistics."""
        rows = [
            {
                "record_id": f"r{index}",
                "rvdi_lcb": float(index),
                "images": [f"/images/{index}.jpg"],
                "prompt": f"question {index}",
                "chosen": f"answer {index}",
            }
            for index in range(9)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "dev_scored.json"
            package = root / "package"
            report = root / "report"
            source.write_text(json.dumps(rows), encoding="utf-8")
            prepare_annotation_package(source, package, sample_size=9, seed=42, strata=3)
            with (package / "control" / "annotation_key.csv").open(encoding="utf-8", newline="") as handle:
                key_rows = list(csv.DictReader(handle))
            labels = {row["record_id"]: row["rvdi_quintile"] for row in key_rows}
            for filename in ("annotator_a.csv", "annotator_b.csv"):
                path = package / "reviewer" / filename
                with path.open(encoding="utf-8", newline="") as handle:
                    annotations = list(csv.DictReader(handle))
                for row in annotations:
                    row["visual_dependence"] = labels[row["record_id"]]
                with path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=annotations[0].keys())
                    writer.writeheader()
                    writer.writerows(annotations)

            summary = analyze_annotation_package(
                package / "control" / "annotation_key.csv",
                package / "reviewer" / "annotator_a.csv",
                package / "reviewer" / "annotator_b.csv",
                report,
                bootstrap_samples=100,
                bootstrap_seed=7,
            )

            self.assertAlmostEqual(summary["weighted_cohen_kappa"], 1.0)
            self.assertGreater(summary["spearman_rho"], 0.9)
            self.assertEqual(summary["sample_size"], 9)
            self.assertTrue((report / "rvdi_human_validation.png").is_file())
            self.assertTrue((report / "validation_summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
