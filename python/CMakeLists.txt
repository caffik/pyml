include(FetchContent)

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
)

target_link_libraries(
    pyml
        PRIVATE svd_classifier_specialization
)

install(TARGETS pyml LIBRARY DESTINATION pyml)
