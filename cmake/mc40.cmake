find_path(MC40_ROOT_DIR
    include/ax_interpreter_external_api.h
    PATHS
    ${PROJECT_SOURCE_DIR}/third_party/mc40/
    $ENV{MC40DIR}
)

if(${MC40_ROOT_DIR} STREQUAL "MC40_ROOT_DIR-NOTFOUND")
    message(FATAL_ERROR "Can not find MC40")
endif()
message(STATUS "Build with MC40 in ${MC40_ROOT_DIR}")

find_path(MC40_INCLUDE_DIR
    ax_interpreter_external_api.h
    PATHS
    ${MC40_ROOT_DIR}/include
    ${INCLUDE_INSTALL_DIR}
)

add_library(libmc40 INTERFACE IMPORTED)
find_library(MC40_LIBRARY
    NAMES libax_interpreter_external.x86.a
    PATHS ${MC40_ROOT_DIR}/lib/)

if(${MC40_LIBRARY} STREQUAL "MC40_LIBRARY-NOTFOUND")
    message(FATAL_ERROR "Can not find MC40 library")
endif()
target_link_libraries(libmc40 INTERFACE ${MC40_LIBRARY})
target_include_directories(libmc40 INTERFACE ${MC40_INCLUDE_DIR})
