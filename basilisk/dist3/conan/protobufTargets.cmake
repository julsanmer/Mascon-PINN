

if(NOT TARGET protobuf::libprotobuf)
    add_library(protobuf::libprotobuf INTERFACE IMPORTED)
endif()

if(NOT TARGET protobuf::libprotoc)
    add_library(protobuf::libprotoc INTERFACE IMPORTED)
endif()

if(NOT TARGET protobuf::protobuf)
    add_library(protobuf::protobuf INTERFACE IMPORTED)
endif()

# Load the debug and release library finders
get_filename_component(_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
file(GLOB CONFIG_FILES "${_DIR}/protobufTarget-*.cmake")

foreach(f ${CONFIG_FILES})
    include(${f})
endforeach()

if(protobuf_FIND_COMPONENTS)
    foreach(_FIND_COMPONENT ${protobuf_FIND_COMPONENTS})
        list(FIND protobuf_COMPONENTS_RELEASE "protobuf::${_FIND_COMPONENT}" _index)
        if(${_index} EQUAL -1)
            conan_message(FATAL_ERROR "Conan: Component '${_FIND_COMPONENT}' NOT found in package 'protobuf'")
        else()
            conan_message(STATUS "Conan: Component '${_FIND_COMPONENT}' found in package 'protobuf'")
        endif()
    endforeach()
endif()