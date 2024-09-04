#include <libml/svd_classification/svd_classifier.hpp>
#include <nanobind/nanobind.h>

namespace nb = nanobind;

void init_setNumThreads(nb::module_ &m) {
  m.def("setNumThreads", &ml::setNumThreads, nb::arg("num_threads"),
    "Sets the number of threads to be used.");
}

void init_getNumThreads(nb::module_ &m) {
  m.def("getNumThreads", &ml::getNumThreads,
        "Gets the number of threads currently set.");
}