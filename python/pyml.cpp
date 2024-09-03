#include <nanobind/nanobind.h>
#include <svd_classifier_specialization/svd_classifier_specialization.hpp>

namespace nb = nanobind;

NB_MODULE(pyml, m) {
  m.doc() = "A small module for machine learning.";
  m.def("setNumThreads", &ml::setNumThreads, nb::arg("num_threads"),
        "Sets the number of threads to be used.");
  m.def("getNumThreads", &ml::getNumThreads,
        "Gets the number of threads currently set.");

  nb::module_ m_svd = m.def_submodule(
      "svd_classifier",
      "SVD classifier module that introduces a class for classification using "
      "SVD. "
      "It also provides functions for calculating vector projections.");

  nb::class_<SVDClassifierSpecialzied>(m_svd, "SVDClassifier")
      .def(nb::init<>())
      .def(nb::init<const matrices_container_type &>())
      .def("fit", &SVDClassifierSpecialzied::fit,
           "Calculates right singular matrices.")
      .def("fit_predict", &SVDClassifierSpecialzied::fit_predict,
           nb::arg("pred_data"), nb::arg("num_of_singulars") = 0,
           "Predicts labels for given data.")
      .def_prop_rw(
          "data", [](SVDClassifierSpecialzied &o) { return o.getData(); },
          [](SVDClassifierSpecialzied &o,
             SVDClassifierSpecialzied::const_reference_data_t data) {
            o.setData(data);
          })
      .def_prop_ro("u_matrices",
                   [](SVDClassifierSpecialzied &o) { return o.getUMatrices(); })
      .def_prop_ro(
          "projections",
          [](SVDClassifierSpecialzied &o) { return o.getProjections(); })
      .doc() =
      "A class for classification using SVD. \n\n "
      "Attributes: \n"
      "---------------- \n"
      "data : list[numpy.ndarray[dtype=float64, shape=(*, *), order='*']] \n "
      "list of matrices, each matrix contains data for specific label. \n"
      "u_matrices : list[numpy.ndarray[dtype=float64, shape=(*, *), "
      "order='*']] \n list of right singular matrices for each matrix in data. "
      "\n"
      "projections : list[numpy.ndarray[dtype=float64, shape=(*, *), "
      "order='*']] \n list of projections for each labels. \n";

  m_svd.def("projection", &projection, nb::arg("from"), nb::arg("onto"),
            "Calculates projection of one matrix onto another.");
  m_svd.def("projections", &projections, nb::arg("from"), nb::arg("onto"),
            nb::arg("span_size") = 0,
            "Calculates projections of one matrix onto a list of matrices.");
}
