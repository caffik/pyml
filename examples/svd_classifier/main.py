import gzip
from mnist import MNIST
import numpy as np
import os
import urllib.request
import shutil
import svd_classifier as svd_c
import time
import pyml
from pyml import svd_classifier as pyml_svd_c


def download_mnist() -> None:
    """Downloads mnist dataset."""

    if os.path.exists("mnist"):
        return None

    os.makedirs("mnist")
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    data_files = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
                  "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]
    for file in data_files:
        url = base_url + file
        urllib.request.urlretrieve(url, f"mnist/{file}")
    return None


def unzip_mnist() -> None:
    """Unzips mnist dataset."""
    data_files = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
                  "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]
    for file in data_files:
        if os.path.exists(f"mnist/{file[:-3]}"):
            continue
        with gzip.open(f"mnist/{file}", "rb") as f_in:
            with open(f"mnist/{file[:-3]}", "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
    return None


if __name__ == "__main__":
    download_mnist()
    unzip_mnist()
    mnist = MNIST("mnist")
    train_images, train_labels = mnist.load_training()
    test_images, test_labels = mnist.load_testing()
    images = np.array(test_images, dtype=np.float64).transpose()

    data_by_label = svd_c.split_data_by(labels=train_labels, data=train_images)
    matrices = [np.array(data_by_label[i], dtype=np.float64).transpose() for i in range(len(data_by_label))]

    start = time.time()
    svd_classification = svd_c.SVDClassification(copy=False)
    svd_classification.fit(matrices=matrices)

    pred_labels = svd_classification.fit_predict(data=images,
                                                 number_of_singular=100)
    duration = time.time() - start
    print(f"Duration: {duration}")
    score = np.mean(pred_labels == np.array(test_labels))
    print(f"Score: {score}")

    pyml.setNumThreads(10)
    start = time.time()
    pyml_svd_classification = pyml_svd_c.SVDClassifier(matrices)
    pyml_svd_classification.fit()
    pyml_pred_labels = pyml_svd_classification.fit_predict(images, 100)
    pyml_duration = time.time() - start
    print(f"Duration: {pyml_duration}")
    pyml_score = np.mean(pyml_pred_labels == np.array(test_labels))
    print(f"Score: {pyml_score}")
