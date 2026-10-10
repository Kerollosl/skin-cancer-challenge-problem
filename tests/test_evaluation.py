import unittest
from evaluation import metrics
class EvaluationTests(unittest.TestCase):
    def test_majority_incorrect(self):self.assertEqual(metrics([1,0,0,0],[1,1,1,1])["accuracy"],.25)
    def test_uniform(self):
        self.assertEqual(metrics([1,1],[1,1])["accuracy"],1)
        self.assertEqual(metrics([0,0],[1,1])["accuracy"],0)
    def test_invalid(self):
        with self.assertRaises(ValueError):metrics([1],[float("nan")])


def test_fractional_labels_rejected():
 import pytest
 with pytest.raises(ValueError):metrics([.5],[.2])

def test_confidence_intervals_are_bounded():
 result=metrics([0,1],[.1,.9])
 assert 0 <= result['accuracy_ci95'][0] < result['accuracy_ci95'][1] <= 1
