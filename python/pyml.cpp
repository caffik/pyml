#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>

namespace nb = nanobind;

void init_setNumThreads(nb::module_ &);
void init_getNumThreads(nb::module_ &);

void init_SVDClassifier(nb::module_ &);
void init_projection(nb::module_ &);
void init_projections(nb::module_ &);

NB_MODULE(pyml, m) {
    m.doc() = "A small module for machine learning.";
    init_setNumThreads(m);
    init_getNumThreads(m);

    nb::module_ m_svd = m.def_submodule(
        "svd_classifier",
        "SVD classifier module that introduces a class for classification using "
        "SVD. "
        "It also provides functions for calculating vector projections.");

    init_SVDClassifier(m_svd);
    init_projection(m_svd);
    init_projections(m_svd);
}
