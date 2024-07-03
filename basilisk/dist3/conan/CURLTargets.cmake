

if(NOT TARGET CURL::libcurl)
    add_library(CURL::libcurl INTERFACE IMPORTED)
endif()

if(NOT TARGET CURL::CURL)
    add_library(CURL::CURL INTERFACE IMPORTED)
endif()

# Load the debug and release library finders
get_filename_component(_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
file(GLOB CONFIG_FILES "${_DIR}/CURLTarget-*.cmake")

foreach(f ${CONFIG_FILES})
    include(${f})
endforeach()

if(CURL_FIND_COMPONENTS)
    foreach(_FIND_COMPONENT ${CURL_FIND_COMPONENTS})
        list(FIND CURL_COMPONENTS_RELEASE "CURL::${_FIND_COMPONENT}" _index)
        if(${_index} EQUAL -1)
            conan_message(FATAL_ERROR "Conan: Component '${_FIND_COMPONENT}' NOT found in package 'CURL'")
        else()
            conan_message(STATUS "Conan: Component '${_FIND_COMPONENT}' found in package 'CURL'")
        endif()
    endforeach()
endif()