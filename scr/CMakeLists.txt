include(FetchContent)

###############
#    libml    #
###############

FetchContent_Declare(
    libml
    GIT_REPOSITORY https://github.com/caffik/libml
    GIT_TAG v1.0.0
)

set(ENABLE_DOCS OFF)
set(ENABLE_TESTING OFF)
FetchContent_MakeAvailable(libml)

###############################################
#   Specialization of libml::svd_classifier   #
###############################################

set(INCLUDE_DIR
    ${PROJECT_SOURCE_DIR}/include
)

add_library(svd_classifier_specialization INTERFACE)

target_link_libraries(svd_classifier_specialization
    INTERFACE
        libml::svd_classifier
)

target_include_directories(svd_classifier_specialization
    INTERFACE
        ${INCLUDE_DIR}
)
