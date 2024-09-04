include(FetchContent)

###############
#    libml    #
###############

FetchContent_Declare(
        libml
        GIT_REPOSITORY https://github.com/caffik/libml
        GIT_TAG        v1.0.0
)

set(ENABLE_DOCS OFF)
set(ENABLE_TESTING OFF)
FetchContent_MakeAvailable(libml)

################
#   nanobind   #
################

FetchContent_Declare(
        nanobind
        GIT_REPOSITORY https://github.com/wjakob/nanobind
        GIT_TAG v2.0.0
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
)

FetchContent_MakeAvailable(nanobind)

######################
#   nanobind Setup   #
######################

nanobind_add_module(pyml
    STABLE_ABI
    NB_STATIC
        pyml.cpp
        cpp/ml.cpp
        cpp/svd_classifier.cpp
        cpp/svd_classification_functions.cpp
)

target_link_libraries(
    pyml
        PRIVATE libml::svd_classification
)

install(TARGETS pyml LIBRARY DESTINATION pyml)
