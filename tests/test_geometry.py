import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GeometryTests(unittest.TestCase):
    def test_centre_corner_iou_is_one_seventh(self):
        module = load("fingerprint")
        for box in ((.1, .2, .3, .4), (.5, .5, .2, .8), (.01, .02, .7, .6)):
            value = module.overlap_over_union(module.centre_read(*box), module.corner_read(*box))
            self.assertAlmostEqual(value, 1 / 7, places=12)

    def test_corner_clipping_has_one_sided_centre_overflow(self):
        module = load("fingerprint")
        stats = module.FieldStats()
        stats.add("leaf", .1, .1, .4, .4, image=(100, 200))
        report = stats.report()
        self.assertTrue(report["one_sided_fingerprint"])
        self.assertEqual(report["centre_read_out_of_frame"]["left"], 1)
        self.assertEqual(report["centre_read_out_of_frame"]["top"], 1)
        self.assertEqual(report["centre_read_out_of_frame"]["right"], 0)
        self.assertEqual(report["centre_read_out_of_frame"]["bottom"], 0)


if __name__ == "__main__":
    unittest.main()
