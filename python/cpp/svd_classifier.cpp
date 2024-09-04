#include <libml/svd_classification/svd_classifier.hpp>
#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

/*
 * Specialization of ml::SVDClassifier.
 */

namespace nb = nanobind;

using matrix_type = nb::DRef<Eigen::MatrixXd>;
using matrices_container_type = std::vector<matrix_type>;
using allocator_type = matrices_container_type::allocator_type;
using value_type = matrices_container_type::value_type;

class SVDClassifierSpecialzied
    : public ml::SVDClassifier<value_type, allocator_type> {
public:
  SVDClassifierSpecialzied(const SVDClassifierSpecialzied &) = delete;
  SVDClassifierSpecialzied &
  operator=(const SVDClassifierSpecialzied &) = delete;
  SVDClassifierSpecialzied(SVDClassifierSpecialzied &&) = delete;
  SVDClassifierSpecialzied &operator=(SVDClassifierSpecialzied &&) = delete;

  SVDClassifierSpecialzied() : SVDClassifier(){};

  explicit SVDClassifierSpecialzied(const matrices_container_type &data)
      : SVDClassifier(data){};

  void fit() { SVDClassifier::fit(); }

  const Eigen::Matrix<std::size_t, Eigen::Dynamic, Eigen::Dynamic> &
  fit_predict(const matrix_type &pred_data,
              const std::size_t number_of_singulars = 0) {
    return SVDClassifier::fit_predict(pred_data, number_of_singulars);
  }
};

void init_SVDClassifier(nb::module_ &m_svd) {
  nb::class_<SVDClassifierSpecialzied>(m_svd, "SVDClassifier")
      .def(nb::init<>())
      .def(nb::init<const std::vector<nb::DRef<Eigen::MatrixXd>> &>())
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
}