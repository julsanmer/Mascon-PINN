
if(NOT TARGET libbacktrace::libbacktrace)
    add_library(libbacktrace::libbacktrace INTERFACE IMPORTED)
endif()

# Load the debug and release library finders
get_filename_component(_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
file(GLOB CONFIG_FILES "${_DIR}/libbacktraceTarget-*.cmake")

foreach(f ${CONFIG_FILES})
    include(${f})
endforeach()
