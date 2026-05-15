import importlib
import unittest


class PdfProcessorOcrAvailabilityTests(unittest.TestCase):
    def setUp(self):
        self.pdf_processor = importlib.import_module("src.processing.pdf.pdf_processor")
        self._original = {
            "HAS_PYTESSERACT": self.pdf_processor.HAS_PYTESSERACT,
            "HAS_TESSERACT_BINARY": self.pdf_processor.HAS_TESSERACT_BINARY,
            "HAS_OCR": self.pdf_processor.HAS_OCR,
        }

    def tearDown(self):
        for name, value in self._original.items():
            setattr(self.pdf_processor, name, value)

    def test_ocr_reason_identifies_missing_python_dependencies(self):
        self.pdf_processor.HAS_PYTESSERACT = False
        self.pdf_processor.HAS_TESSERACT_BINARY = False
        self.pdf_processor.HAS_OCR = False

        self.assertEqual(
            self.pdf_processor._ocr_unavailable_reason(),
            "pytesseract package or Pillow is not installed",
        )

    def test_ocr_reason_identifies_missing_tesseract_binary(self):
        self.pdf_processor.HAS_PYTESSERACT = True
        self.pdf_processor.HAS_TESSERACT_BINARY = False
        self.pdf_processor.HAS_OCR = False

        self.assertEqual(
            self.pdf_processor._ocr_unavailable_reason(),
            "tesseract binary is not installed or not on PATH",
        )

    def test_ocr_reason_confirms_available_ocr_stack(self):
        self.pdf_processor.HAS_PYTESSERACT = True
        self.pdf_processor.HAS_TESSERACT_BINARY = True
        self.pdf_processor.HAS_OCR = True

        self.assertEqual(
            self.pdf_processor._ocr_unavailable_reason(),
            "OCR support is available",
        )


if __name__ == "__main__":
    unittest.main()
