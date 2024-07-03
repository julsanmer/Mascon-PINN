

if(NOT TARGET Microsoft.GSL::GSL)
    add_library(Microsoft.GSL::GSL INTERFACE IMPORTED)
endif()

if(NOT TARGET Microsoft.GSL::Microsoft.GSL)
    add_library(Microsoft.GSL::Microsoft.GSL INTERFACE IMPORTED)
endif()

# Load the debug and release library finders
get_filename_component(_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
file(GLOB CONFIG_FILES "${_DIR}/Microsoft.GSLTarget-*.cmake")

foreach(f ${CONFIG_FILES})
    include(${f})
endforeach()

if(Microsoft.GSL_FIND_COMPONENTS)
    foreach(_FIND_COMPONENT ${Microsoft.GSL_FIND_COMPONENTS})
        list(FIND Microsoft.GSL_COMPONENTS_RELEASE "Microsoft.GSL::${_FIND_COMPONENT}" _index)
        if(${_index} EQUAL -1)
            conan_message(FATAL_ERROR "Conan: Component '${_FIND_COMPONENT}' NOT found in package 'Microsoft.GSL'")
        else()
            conan_message(STATUS "Conan: Component '${_FIND_COMPONENT}' found in package 'Microsoft.GSL'")
        endif()
    endforeach()
endif()