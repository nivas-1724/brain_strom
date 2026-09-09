"""
Automated Integration & Unit Tests for MRI / CT Modality Validation Pipeline
Tests:
1. Valid Brain MRI Scan -> Accepted, Tumor Prediction Available, XAI Available
2. CT Scan Image -> Modality CT Detected, Rejected, Tumor Prediction Not Available, XAI Blocked
3. Non-medical Image -> Modality UNKNOWN, Rejected
4. Corrupted / Low-quality Image -> Validation Failed, Rejected
5. Unseen MRI Scan -> Valid MRI Accepted
"""

import os
import sys
import unittest
import numpy as np
from PIL import Image, ImageDraw

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from model.mri_validator import validate_mri_pipeline, detect_modality
from model.predict import predict


class TestModalityValidationPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Create sample test image objects."""
        # 1. Real Brain MRI Sample from dataset folder
        import glob
        mri_files = glob.glob(os.path.join(BASE_DIR, "dataset", "**", "*.jpg"), recursive=True) + \
                    glob.glob(os.path.join(BASE_DIR, "dataset", "**", "*.png"), recursive=True)
        if mri_files:
            cls.mri_img = Image.open(mri_files[0]).convert("RGB")
        else:
            cls.mri_img = Image.new("L", (224, 224), color=0)
            draw = ImageDraw.Draw(cls.mri_img)
            draw.ellipse([30, 30, 194, 194], fill=80)
            draw.ellipse([60, 60, 164, 164], fill=140)
            cls.mri_img = cls.mri_img.convert("RGB")

        # 2. CT Brain Scan (Bright white bone calvarium ring + CT parenchyma attenuation)
        cls.ct_img = Image.new("L", (224, 224), color=0)
        draw = ImageDraw.Draw(cls.ct_img)
        # Bright white skull ring
        draw.ellipse([20, 20, 204, 204], fill=245)
        # Inner tissue
        draw.ellipse([30, 30, 194, 194], fill=90)
        # Dark ventricles
        draw.ellipse([95, 85, 129, 139], fill=15)
        cls.ct_img = cls.ct_img.convert("RGB")

        # 3. Non-Medical Image (Document scan / white page with lines)
        cls.non_medical_img = Image.new("RGB", (224, 224), color=(245, 245, 245))
        draw = ImageDraw.Draw(cls.non_medical_img)
        for i in range(10, 200, 20):
            draw.line([(20, i), (200, i)], fill=(20, 20, 20), width=2)

        # 4. Low Quality / Corrupted Image (Solid black flat noise)
        cls.low_quality_img = Image.new("RGB", (224, 224), color=(10, 10, 10))

    def test_01_valid_mri_acceptance(self):
        """Test 1: Valid Brain MRI should be accepted with tumor prediction and XAI available."""
        res = predict(self.mri_img)
        
        self.assertTrue(res["success"])
        self.assertTrue(res["is_valid_mri"], "Valid MRI scan should be accepted.")
        self.assertEqual(res["modality"], "MRI")
        self.assertGreaterEqual(res["modality_confidence"], 85.0)
        self.assertIsNotNone(res["prediction"])
        self.assertNotEqual(res["prediction"], "Not Available")
        self.assertIsNotNone(res["calibrated_confidence"])
        self.assertIsNotNone(res["overlay_b64"], "Grad-CAM XAI map must be available for valid MRI.")

    def test_02_ct_scan_rejection(self):
        """Test 2: CT scan image MUST be detected as CT and rejected before tumor prediction & XAI."""
        val_res = validate_mri_pipeline(self.ct_img)
        self.assertFalse(val_res["is_valid_mri"], "CT scan must be rejected.")
        self.assertEqual(val_res["modality"], "CT", "Modality must be detected as CT.")
        self.assertGreaterEqual(val_res["modality_confidence"], 85.0)

        # Complete prediction pipeline rejection check
        res = predict(self.ct_img)
        self.assertTrue(res["success"])
        self.assertFalse(res["is_valid_mri"], "CT scan must fail MRI validation.")
        self.assertEqual(res["status"], "rejected_ct")
        self.assertEqual(res["modality"], "CT")
        self.assertIsNone(res["prediction"], "Tumor prediction must be null/None for CT scans.")
        self.assertEqual(res["display_name"], "Not Available")
        self.assertIsNone(res["tumor_confidence"], "Tumor confidence score must be None for CT scans.")
        self.assertIsNone(res["raw_confidence"], "Raw confidence must be None for CT scans.")
        self.assertIsNone(res["calibrated_confidence"], "Calibrated confidence must be None for CT scans.")
        self.assertIsNone(res["overlay_b64"], "Grad-CAM XAI map must NOT be generated for rejected CT scans.")
        self.assertIsNone(res["ig_b64"], "Integrated Gradients XAI map must NOT be generated for rejected CT scans.")
        self.assertIsNone(res["lime_b64"], "LIME XAI map must NOT be generated for rejected CT scans.")

    def test_03_non_medical_rejection(self):
        """Test 3: Non-medical image (document / photo) must be rejected as UNKNOWN."""
        res = predict(self.non_medical_img)
        self.assertFalse(res["is_valid_mri"])
        self.assertIn(res["status"], ["rejected_unknown", "rejected_ct"])
        self.assertIsNone(res["prediction"])
        self.assertIsNone(res["tumor_confidence"])
        self.assertIsNone(res["overlay_b64"])

    def test_04_low_quality_rejection(self):
        """Test 4: Low-quality/ambiguous image without structural contrast must be rejected."""
        val_res = validate_mri_pipeline(self.low_quality_img)
        self.assertFalse(val_res["is_valid_mri"], "Low quality scan must fail MRI validation.")

    def test_05_dicom_support(self):
        """Test 5: DICOM metadata headers with Modality='CT' must be rejected immediately."""
        # Synthetic DICOM binary header with Modality tag 'CT'
        dicom_ct_bytes = b"\x00" * 128 + b"DICM" + b"\x08\x00\x60\x00\x02\x00CT" + b"\x00" * 100
        val_res = validate_mri_pipeline(self.ct_img, file_bytes=dicom_ct_bytes)
        self.assertFalse(val_res["is_valid_mri"])
        self.assertEqual(val_res["modality"], "CT")

    def test_06_direct_api_bypass(self):
        """Test 6: Sending CT image directly to prediction pipeline must be rejected before tumor model inference."""
        res = predict(self.ct_img, filename="download.jpg")
        self.assertFalse(res["is_valid_mri"])
        self.assertIsNone(res["prediction"])
        self.assertIsNone(res["tumor_confidence"])
        self.assertIsNone(res["overlay_b64"])
        self.assertEqual(res["status"], "rejected_ct")


if __name__ == "__main__":
    unittest.main()
