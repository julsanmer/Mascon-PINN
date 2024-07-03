########## MACROS ###########################################################################
#############################################################################################

function(conan_message MESSAGE_OUTPUT)
    if(NOT CONAN_CMAKE_SILENT_OUTPUT)
        message(${ARGV${0}})
    endif()
endfunction()


# Requires CMake > 3.0
if(${CMAKE_VERSION} VERSION_LESS "3.0")
    message(FATAL_ERROR "The 'cmake_find_package_multi' generator only works with CMake > 3.0")
endif()

include(${CMAKE_CURRENT_LIST_DIR}/ONNXTargets.cmake)

########## FIND PACKAGE DEPENDENCY ##########################################################
#############################################################################################

include(CMakeFindDependencyMacro)

if(NOT protobuf_FOUND)
    if(${CMAKE_VERSION} VERSION_LESS "3.9.0")
        find_package(protobuf REQUIRED NO_MODULE)
    else()
        find_dependency(protobuf REQUIRED NO_MODULE)
    endif()
else()
    message(STATUS "Dependency protobuf already found")
endif()

########## TARGETS PROPERTIES ###############################################################
#############################################################################################

########## COMPONENT onnx_proto TARGET PROPERTIES ######################################

set_property(TARGET ONNX::onnx_proto PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${ONNX_onnx_proto_LINK_LIBS_DEBUG}
                ${ONNX_onnx_proto_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${ONNX_onnx_proto_LINK_LIBS_RELEASE}
                ${ONNX_onnx_proto_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${ONNX_onnx_proto_LINK_LIBS_RELWITHDEBINFO}
                ${ONNX_onnx_proto_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${ONNX_onnx_proto_LINK_LIBS_MINSIZEREL}
                ${ONNX_onnx_proto_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET ONNX::onnx_proto PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${ONNX_onnx_proto_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${ONNX_onnx_proto_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${ONNX_onnx_proto_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${ONNX_onnx_proto_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET ONNX::onnx_proto PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${ONNX_onnx_proto_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${ONNX_onnx_proto_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${ONNX_onnx_proto_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${ONNX_onnx_proto_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET ONNX::onnx_proto PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${ONNX_onnx_proto_COMPILE_OPTIONS_C_DEBUG}
                 ${ONNX_onnx_proto_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${ONNX_onnx_proto_COMPILE_OPTIONS_C_RELEASE}
                 ${ONNX_onnx_proto_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${ONNX_onnx_proto_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${ONNX_onnx_proto_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${ONNX_onnx_proto_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${ONNX_onnx_proto_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(ONNX_onnx_proto_TARGET_PROPERTIES TRUE)

########## COMPONENT onnx TARGET PROPERTIES ######################################

set_property(TARGET ONNX::onnx PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${ONNX_onnx_LINK_LIBS_DEBUG}
                ${ONNX_onnx_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${ONNX_onnx_LINK_LIBS_RELEASE}
                ${ONNX_onnx_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${ONNX_onnx_LINK_LIBS_RELWITHDEBINFO}
                ${ONNX_onnx_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${ONNX_onnx_LINK_LIBS_MINSIZEREL}
                ${ONNX_onnx_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET ONNX::onnx PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${ONNX_onnx_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${ONNX_onnx_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${ONNX_onnx_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${ONNX_onnx_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET ONNX::onnx PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${ONNX_onnx_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${ONNX_onnx_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${ONNX_onnx_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${ONNX_onnx_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET ONNX::onnx PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${ONNX_onnx_COMPILE_OPTIONS_C_DEBUG}
                 ${ONNX_onnx_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${ONNX_onnx_COMPILE_OPTIONS_C_RELEASE}
                 ${ONNX_onnx_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${ONNX_onnx_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${ONNX_onnx_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${ONNX_onnx_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${ONNX_onnx_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(ONNX_onnx_TARGET_PROPERTIES TRUE)

########## GLOBAL TARGET PROPERTIES #########################################################

if(NOT ONNX_ONNX_TARGET_PROPERTIES)
    set_property(TARGET ONNX::ONNX APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                 $<$<CONFIG:Debug>:${ONNX_COMPONENTS_DEBUG}>
                 $<$<CONFIG:Release>:${ONNX_COMPONENTS_RELEASE}>
                 $<$<CONFIG:RelWithDebInfo>:${ONNX_COMPONENTS_RELWITHDEBINFO}>
                 $<$<CONFIG:MinSizeRel>:${ONNX_COMPONENTS_MINSIZEREL}>)
endif()

########## BUILD MODULES ####################################################################
#############################################################################################

########## COMPONENT onnx_proto BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${ONNX_onnx_proto_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${ONNX_onnx_proto_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${ONNX_onnx_proto_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${ONNX_onnx_proto_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT onnx BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${ONNX_onnx_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${ONNX_onnx_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${ONNX_onnx_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${ONNX_onnx_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()