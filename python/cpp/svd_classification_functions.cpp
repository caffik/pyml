#include <libml/svd_classification/projection.hpp>
#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

using matrix_type = nb::DRef<Eigen::MatrixXd>;
using matrices_container_type = std::vector<matrix_type>;

Eigen::MatrixXd projection(const matrix_type &from,
                                  const matrix_type &onto) {
  return ml::projection(from, onto);
}

std::vector<Eigen::MatrixXd>
projections(const matrix_type &from, const matrices_container_type &onto,
            const std::size_t span_size = 0) {
  return ml::projections(from, onto, span_size);
}

void init_projection(nb::module_ &m_svd) {
  m_svd.def("projection", projection, nb::arg("from"), nb::arg("onto"),
            "Calculates projection of one matrix onto another.");
}

void init_projections(nb::module_ &m_svd) {
  m_svd.def("projections", projections, nb::arg("from"), nb::arg("onto"),
            nb::arg("span_size") = 0,
            "Calculates projections of one matrix onto a list of matrices.");
}
