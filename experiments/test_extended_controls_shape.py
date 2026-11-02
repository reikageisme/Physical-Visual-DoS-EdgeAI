"""Regression fixture for retaining batch axes in integrity-control placement."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from extended_controls import integrity


class IntegrityShapeFixture(unittest.TestCase):
    def test_actual_location_search_receives_single_image_batches(self):
        train = torch.zeros(4,3,320,320)
        with tempfile.TemporaryDirectory(prefix='sponge-shape-fixture-') as directory:
            folder = Path(directory)
            # Actual OpenCV location search runs. Stop immediately afterward,
            # before EOT, gradients, model forwards or artifact generation.
            with patch('extended_controls.make_eot',side_effect=RuntimeError('fixture-stop-after-locations')):
                with self.assertRaisesRegex(RuntimeError,'fixture-stop-after-locations'):
                    integrity(None,train,0,folder)
            self.assertEqual(list(folder.iterdir()),[])


if __name__ == '__main__':
    unittest.main()
