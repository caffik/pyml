import numpy as np
from copy import deepcopy
from typing import Sequence, Union


def split_data_by(data: Sequence, labels: Sequence) -> dict:
    """Returns a dictionary containing data partitioned by labels.

    :param data: Sequence of elements of any type.
    :param labels: Sequence of hashable objects, where the ith object must be a label of the ith element of the data.
    :return: Dictionary containing data partitioned by labels.

    :Example:
    data = [1,2,3,4,5]
    labels = [0,0,0,1,1]
    >>> split_data_by(data=data, labels=labels)
    {0: [1, 2, 3], 1: [4, 5]}
    """
    result = {label: list() for label in labels}
    for i, label in enumerate(labels):
        result[label].append(data[i])
    return result


class SVDClassification:
    """Classifier based on singular value decomposition.

    Attributes:
    ----------
    :param copy: `True` if provided data should be copied, defaults `False`
    :param matrices: List with data divided by labels.
    :param u_matrices: List containing right singular vectors corresponding to matrices.
    """
    def __init__(self, copy=True):
        self.copy: bool = copy
        self.matrices: Union[None, list[np.ndarray]] = None
        self.u_matrices: Union[None, list[np.ndarray]] = None
        self.__projections: Union[None, list[np.ndarray]] = None
        self.__distances: Union[None, list[np.array]] = None
        self.__pred_labels: Union[None, list[int]] = None

    def fit(self, matrices: list[np.ndarray]) -> None:
        """Calculates right singular matrix for each matrix in matrices

        :param matrices:
        :return: None
        """
        self.matrices = matrices if not self.copy else deepcopy(matrices)
        self.u_matrices = [np.linalg.svd(matrix, full_matrices=False)[0] for matrix in matrices]

    @staticmethod
    def projection_mapping(data: np.ndarray, onto: np.ndarray) -> np.ndarray:
        """Returns the matrix that containing the projections of the columns of `data` onto the space spanned by
        columns of `onto`.

        :param data: Matrix to be projected.
        :param onto: Matrix composed by a row vectors that spans space.
        :return: Projection of vectors.

        :Example:
        data = np.array([[1, 0],
                         [0, 1]])
        onto = np.array([[1, -1]
                         [1, 1]])
        >>> projection_mapping(data=data, onto=onto)
        array([[2., 0.],
               [0., 2.]])
        """
        data_transposed = data.transpose()
        dot = data_transposed @ onto
        projection = np.ndarray(shape=(data.shape[0], data.shape[1]), dtype=np.float64)
        for i in range(data.shape[1]):
            projection[:, i] = np.sum(onto * dot[i], axis=1)
        return projection

    @staticmethod
    def distance(a: np.ndarray, b: np.ndarray) -> np.array:
        """Returns Euclidean distance between columns of matrices.

        :param a: Matrix composed with row vectors if only row vector is provided then it is broadcasted to b shape.
        :param b: Matrix composed with row vectors if only row vector is provided then it is broadcasted to a shape.
        :return: Array consisting distances between corresponding vectors.
        """
        return np.linalg.norm(a - b, axis=0)

    def _get_labels(self) -> None:
        self.__pred_labels = np.argmin(self.__distances, axis=0).tolist()
        return None

    def fit_predict(self, data: np.ndarray, number_of_singular: Union[int, None] = None) -> np.array:
        """Predicts labels of given data.

        :param data: Matrix composed of row vectors data.
        :param number_of_singular: Number of singular vectors to be used in projection.
        :return: Predicted labels.
        """
        if self.__projections is None:
            self.__projections = list()
            for u in self.u_matrices:
                cols = u.shape[1] if number_of_singular is None else min(number_of_singular, u.shape[1])
                self.__projections.append(self.projection_mapping(data, u[:, 0:cols]))

        if self.__distances is None:
            self.__distances = list()
            for projection in self.__projections:
                self.__distances.append(self.distance(data, projection))
            self._get_labels()
        return self.__pred_labels
