

if(NOT TARGET Eigen3::Eigen)
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
endif()

if(NOT TARGET Eigen3::Eigen3)
    add_library(Eigen3::Eigen3 INTERFACE IMPORTED)
endif()

# Load the debug and release library finders
get_filename_component(_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
file(GLOB CONFIG_FILES "${_DIR}/Eigen3Target-*.cmake")

foreach(f ${CONFIG_FILES})
    include(${f})
endforeach()

if(Eigen3_FIND_COMPONENTS)
    foreach(_FIND_COMPONENT ${Eigen3_FIND_COMPONENTS})
        list(FIND Eigen3_COMPONENTS_RELEASE "Eigen3::${_FIND_COMPONENT}" _index)
        if(${_index} EQUAL -1)
            conan_message(FATAL_ERROR "Conan: Component '${_FIND_COMPONENT}' NOT found in package 'Eigen3'")
        else()
            conan_message(STATUS "Conan: Component '${_FIND_COMPONENT}' found in package 'Eigen3'")
        endif()
    endforeach()
endif()