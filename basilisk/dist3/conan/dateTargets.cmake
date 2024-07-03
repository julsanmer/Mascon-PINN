

if(NOT TARGET date::date-tz)
    add_library(date::date-tz INTERFACE IMPORTED)
endif()

if(NOT TARGET date::date)
    add_library(date::date INTERFACE IMPORTED)
endif()

# Load the debug and release library finders
get_filename_component(_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
file(GLOB CONFIG_FILES "${_DIR}/dateTarget-*.cmake")

foreach(f ${CONFIG_FILES})
    include(${f})
endforeach()

if(date_FIND_COMPONENTS)
    foreach(_FIND_COMPONENT ${date_FIND_COMPONENTS})
        list(FIND date_COMPONENTS_RELEASE "date::${_FIND_COMPONENT}" _index)
        if(${_index} EQUAL -1)
            conan_message(FATAL_ERROR "Conan: Component '${_FIND_COMPONENT}' NOT found in package 'date'")
        else()
            conan_message(STATUS "Conan: Component '${_FIND_COMPONENT}' found in package 'date'")
        endif()
    endforeach()
endif()