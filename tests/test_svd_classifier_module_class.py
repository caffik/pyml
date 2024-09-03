import numpy as np
import unittest
from pyml import svd_classifier


class SVDModuleClass(unittest.TestCase):
    def test_constructor(self):
        data = [np.random.rand(3, 2).astype(np.float64), np.random.rand(3, 3).astype(np.float64)]
        model = svd_classifier.SVDClassifier(data)

        for data, model_data in zip(data, model.data):
            self.assertTrue(np.array_equal(data, model_data))

    def test_fit(self):
        data = [np.random.rand(3, 3).astype(np.float64), np.random.rand(3, 3).astype(np.float64)]
        model = svd_classifier.SVDClassifier(data)
        model.fit()

        for u_matrix in model.u_matrices:
            self.assertEqual(u_matrix.shape, (3, 3))

    def test_fit_predict(self):
        data = [np.random.rand(3, 3).astype(np.float64), np.random.rand(3, 3).astype(np.float64)]
        model = svd_classifier.SVDClassifier(data)
        model.fit()

        x = np.random.rand(3, 10).astype(np.float64)
        prediction = model.fit_predict(x)

        self.assertEqual(prediction.shape, (1, 10))

        for value in prediction[0]:
            self.assertTrue(value in [0, 1])

    def test_projections(self):
        data = [np.random.rand(3, 3).astype(np.float64), np.random.rand(3, 3).astype(np.float64)]
        model = svd_classifier.SVDClassifier(data)
        model.fit()

        x = np.random.rand(3, 10).astype(np.float64)
        prediction = model.fit_predict(x)

        self.assertEqual(len(model.projections), 2)

        for projection in model.projections:
            self.assertEqual(projection.shape, (3, 10))


if __name__ == '__main__':
    unittest.main()
