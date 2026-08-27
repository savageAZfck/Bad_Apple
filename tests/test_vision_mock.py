"""Mock-based tests for badapple_vision without loading a VLM."""
import unittest
from unittest import mock

import badapple_vision


class TestVisionHostWithoutModel(unittest.TestCase):
    def tearDown(self) -> None:
        badapple_vision._vision_host = None

    @mock.patch("badapple_vision._vlm_load")
    def test_describe_uses_loaded_model(self, mock_load) -> None:
        mock_load.return_value = (mock.MagicMock(), mock.MagicMock())
        host = badapple_vision.VisionHost()
        host._load()
        mock_load.assert_called_once()


if __name__ == "__main__":
    unittest.main()
