#ifndef SVD_CLASSIFIER_SPEC_H
#define SVD_CLASSIFIER_SPEC_H

#include <libml/svd_classifier/svd_classifier.hpp>
#include <nanobind/stl/vector.h>
#include <nanobind/eigen/dense.h>

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

  SVDClassifierSpecialzied()
      : SVDClassifier<
            Eigen::Ref<Eigen::Matrix<double, -1, -1>, 0, Eigen::Stride<-1, -1>>,
            std::allocator<Eigen::Ref<Eigen::Matrix<double, -1, -1>, 0,
                                      Eigen::Stride<-1, -1>>>>(){};

  explicit SVDClassifierSpecialzied(const matrices_container_type &data)
      : ml::SVDClassifier<
            Eigen::Ref<Eigen::Matrix<double, -1, -1>, 0, Eigen::Stride<-1, -1>>,
            std::allocator<Eigen::Ref<Eigen::Matrix<double, -1, -1>, 0,
                                      Eigen::Stride<-1, -1>>>>(data){};

  void fit() { ml::SVDClassifier<value_type, allocator_type>::fit(); }

  const Eigen::Matrix<std::size_t, Eigen::Dynamic, Eigen::Dynamic> &
  fit_predict(const matrix_type &pred_data,
              const std::size_t number_of_singulars = 0) {
    return ml::SVDClassifier<value_type, allocator_type>::fit_predict(
        pred_data, number_of_singulars);
  }
};

inline Eigen::MatrixXd projection(const matrix_type &from, const matrix_type &onto) {
  return ml::projection(from, onto);
}

inline std::vector<Eigen::MatrixXd> projections(
    const matrix_type &from, const matrices_container_type &onto,
    const std::size_t span_size = 0) {
  return ml::projections(from, onto, span_size);
}

#endif // SVD_CLASSIFIER_SPEC_H
