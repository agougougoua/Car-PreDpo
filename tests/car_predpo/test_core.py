import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops

from car_predpo.audit import annotate_capabilities, length_features, stratified_group_split, validate_record
from car_predpo.margin import build_margins
from car_predpo.perturb import make_perturbations
from car_predpo.rvdi import compute_rvdi


class AuditTests(unittest.TestCase):
    def test_validate_record_accepts_readable_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "sample.jpg"
            Image.new("RGB", (24, 24), "red").save(image)
            row = {
                "prompt": "<image>\nWhere is the red cube?",
                "chosen": "It is left of the ball.",
                "rejected": "It is right of the ball.",
                "images": [str(image)],
            }
            self.assertEqual(validate_record(row), [])

    def test_validate_record_reports_invalid_preference(self):
        row = {"prompt": "x", "chosen": "same", "rejected": "same", "images": []}
        errors = validate_record(row)
        self.assertIn("chosen_equals_rejected", errors)
        self.assertIn("images_empty", errors)

    def test_capability_and_length_annotations_are_auditable(self):
        labels, evidence = annotate_capabilities(
            "Where is the cat?",
            "The cat is left of and above the dog.",
            "The cat is right of and below the dog.",
        )
        self.assertIn("relation_spatial", labels)
        self.assertTrue(evidence["relation_spatial"])
        lengths = length_features("short", "a much longer rejected answer", lambda s: len(s.split()) + 1)
        self.assertGreater(lengths["length_risk"], 0.5)
        self.assertEqual(lengths["chosen_token_length"], 2)

    def test_group_split_is_deterministic_and_has_no_image_overlap(self):
        rows = [
            {"record_id": f"r{i}", "image_key": f"im{i // 2}", "capabilities": [f"c{i % 2}"]}
            for i in range(20)
        ]
        first = stratified_group_split(rows, dev_fraction=0.2, seed=20260723)
        second = stratified_group_split(rows, dev_fraction=0.2, seed=20260723)
        self.assertEqual(first, second)
        train_images = {r["image_key"] for r in rows if first[r["record_id"]] == "train"}
        dev_images = {r["image_key"] for r in rows if first[r["record_id"]] == "dev"}
        self.assertFalse(train_images & dev_images)
        self.assertGreater(len(dev_images), 0)


class PerturbationTests(unittest.TestCase):
    def test_four_perturbations_preserve_dimensions_and_change_pixels(self):
        x = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)
        image = Image.fromarray(x, "RGB")
        outputs, metadata = make_perturbations(image)
        self.assertEqual(list(outputs), ["gaussian_blur", "resize_restore", "jpeg", "brightness_contrast"])
        self.assertEqual(set(outputs), set(metadata))
        for changed in outputs.values():
            self.assertEqual(changed.size, image.size)
            self.assertIsNotNone(ImageChops.difference(image, changed).getbbox())


class RvdiTests(unittest.TestCase):
    def test_rvdi_uses_population_std_and_lcb(self):
        result = compute_rvdi([1.0, 2.0, 3.0, 4.0], gamma=0.5)
        self.assertAlmostEqual(result["rvdi_mean"], 2.5)
        self.assertAlmostEqual(result["rvdi_std"], math.sqrt(1.25))
        self.assertAlmostEqual(result["rvdi_lcb"], 2.5 - 0.5 * math.sqrt(1.25))


class MarginTests(unittest.TestCase):
    def test_margin_formula_and_distribution_guard(self):
        rows = [
            {
                "percentile": p,
                "rvdi_std_normalized": 0.2,
                "relation_confidence": 1.0 if i % 2 else 0.0,
                "length_risk": 0.1,
                "delta_h": 2.0,
            }
            for i, p in enumerate(np.linspace(0.05, 0.95, 20))
        ]
        enriched, report = build_margins(rows)
        self.assertEqual(len(enriched), 20)
        self.assertTrue(all(0.03 <= row["lambda_i"] <= 0.20 for row in enriched))
        self.assertTrue(all(row["margin"] == row["lambda_i"] * 2.0 for row in enriched))
        self.assertLessEqual(report["clip_fraction_any"], 0.10)
        self.assertTrue(report["valid_for_full_training"])


if __name__ == "__main__":
    unittest.main()
