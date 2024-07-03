

if(NOT TARGET flatbuffers::flatbuffers)
    add_library(flatbuffers::flatbuffers INTERFACE IMPORTED)
endif()

if(NOT TARGET flatbuffers::flatbuffers)
    add_library(flatbuffers::flatbuffers INTERFACE IMPORTED)
endif()

# Load the debug and release library finders
get_filename_component(_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
file(GLOB CONFIG_FILES "${_DIR}/flatbuffersTarget-*.cmake")

foreach(f ${CONFIG_FILES})
    include(${f})
endforeach()

if(flatbuffers_FIND_COMPONENTS)
    foreach(_FIND_COMPONENT ${flatbuffers_FIND_COMPONENTS})
        list(FIND flatbuffers_COMPONENTS_RELEASE "flatbuffers::${_FIND_COMPONENT}" _index)
        if(${_index} EQUAL -1)
            conan_message(FATAL_ERROR "Conan: Component '${_FIND_COMPONENT}' NOT found in package 'flatbuffers'")
        else()
            conan_message(STATUS "Conan: Component '${_FIND_COMPONENT}' found in package 'flatbuffers'")
        endif()
    endforeach()
endif()