

if(NOT TARGET ONNX::onnx_proto)
    add_library(ONNX::onnx_proto INTERFACE IMPORTED)
endif()

if(NOT TARGET ONNX::onnx)
    add_library(ONNX::onnx INTERFACE IMPORTED)
endif()

if(NOT TARGET ONNX::ONNX)
    add_library(ONNX::ONNX INTERFACE IMPORTED)
endif()

# Load the debug and release library finders
get_filename_component(_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
file(GLOB CONFIG_FILES "${_DIR}/ONNXTarget-*.cmake")

foreach(f ${CONFIG_FILES})
    include(${f})
endforeach()

if(ONNX_FIND_COMPONENTS)
    foreach(_FIND_COMPONENT ${ONNX_FIND_COMPONENTS})
        list(FIND ONNX_COMPONENTS_RELEASE "ONNX::${_FIND_COMPONENT}" _index)
        if(${_index} EQUAL -1)
            conan_message(FATAL_ERROR "Conan: Component '${_FIND_COMPONENT}' NOT found in package 'ONNX'")
        else()
            conan_message(STATUS "Conan: Component '${_FIND_COMPONENT}' found in package 'ONNX'")
        endif()
    endforeach()
endif()