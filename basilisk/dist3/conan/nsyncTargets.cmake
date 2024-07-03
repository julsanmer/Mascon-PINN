

if(NOT TARGET nsync::nsync_cpp)
    add_library(nsync::nsync_cpp INTERFACE IMPORTED)
endif()

if(NOT TARGET nsync::nsync_c)
    add_library(nsync::nsync_c INTERFACE IMPORTED)
endif()

if(NOT TARGET nsync::nsync)
    add_library(nsync::nsync INTERFACE IMPORTED)
endif()

# Load the debug and release library finders
get_filename_component(_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
file(GLOB CONFIG_FILES "${_DIR}/nsyncTarget-*.cmake")

foreach(f ${CONFIG_FILES})
    include(${f})
endforeach()

if(nsync_FIND_COMPONENTS)
    foreach(_FIND_COMPONENT ${nsync_FIND_COMPONENTS})
        list(FIND nsync_COMPONENTS_RELEASE "nsync::${_FIND_COMPONENT}" _index)
        if(${_index} EQUAL -1)
            conan_message(FATAL_ERROR "Conan: Component '${_FIND_COMPONENT}' NOT found in package 'nsync'")
        else()
            conan_message(STATUS "Conan: Component '${_FIND_COMPONENT}' found in package 'nsync'")
        endif()
    endforeach()
endif()