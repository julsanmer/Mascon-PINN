

if(NOT TARGET cpuinfo::clog)
    add_library(cpuinfo::clog INTERFACE IMPORTED)
endif()

if(NOT TARGET cpuinfo::cpuinfo)
    add_library(cpuinfo::cpuinfo INTERFACE IMPORTED)
endif()

if(NOT TARGET cpuinfo::cpuinfo)
    add_library(cpuinfo::cpuinfo INTERFACE IMPORTED)
endif()

# Load the debug and release library finders
get_filename_component(_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
file(GLOB CONFIG_FILES "${_DIR}/cpuinfoTarget-*.cmake")

foreach(f ${CONFIG_FILES})
    include(${f})
endforeach()

if(cpuinfo_FIND_COMPONENTS)
    foreach(_FIND_COMPONENT ${cpuinfo_FIND_COMPONENTS})
        list(FIND cpuinfo_COMPONENTS_RELEASE "cpuinfo::${_FIND_COMPONENT}" _index)
        if(${_index} EQUAL -1)
            conan_message(FATAL_ERROR "Conan: Component '${_FIND_COMPONENT}' NOT found in package 'cpuinfo'")
        else()
            conan_message(STATUS "Conan: Component '${_FIND_COMPONENT}' found in package 'cpuinfo'")
        endif()
    endforeach()
endif()