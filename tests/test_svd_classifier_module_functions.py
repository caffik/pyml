import numpy as np
import unittest
from pyml import svd_classifier


class SVDModuleFunctions(unittest.TestCase):
    def test_projection(self):
        x = np.array([[1, 2],
                      [3, 4],
                      [5, 6]], dtype=np.float64)

        on = np.array([[1, 0],
                       [0, 0],
                       [0, 1]], dtype=np.float64)

        projection = svd_classifier.projection(x, on)
        expected = np.array([[1, 2],
                             [0, 0],
                             [5, 6]], dtype=np.float64)
        self.assertTrue(np.array_equal(projection, expected))

    def test_projections(self):
        x = np.array([[1, 2],
                      [3, 4],
                      [5, 6]], dtype=np.float64)

        on = [np.array([[1, 0], [0, 0], [0, 1]], dtype=np.float64),
              np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float64)]

        projections = svd_classifier.projections(x, on)
        expected = [np.array([[1, 2], [0, 0], [5, 6]], dtype=np.float64),
                    np.array([[0, 0], [3, 4], [5, 6]], dtype=np.float64)]

        for projection, expected in zip(projections, expected):
            self.assertTrue(np.array_equal(projection, expected))


if __name__ == '__main__':
    unittest.main()
