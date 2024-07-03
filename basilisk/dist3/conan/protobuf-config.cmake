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

include(${CMAKE_CURRENT_LIST_DIR}/protobufTargets.cmake)

########## FIND PACKAGE DEPENDENCY ##########################################################
#############################################################################################

include(CMakeFindDependencyMacro)

if(NOT ZLIB_FOUND)
    if(${CMAKE_VERSION} VERSION_LESS "3.9.0")
        find_package(ZLIB REQUIRED NO_MODULE)
    else()
        find_dependency(ZLIB REQUIRED NO_MODULE)
    endif()
else()
    message(STATUS "Dependency ZLIB already found")
endif()

########## TARGETS PROPERTIES ###############################################################
#############################################################################################

########## COMPONENT libprotobuf TARGET PROPERTIES ######################################

set_property(TARGET protobuf::libprotobuf PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${protobuf_libprotobuf_LINK_LIBS_DEBUG}
                ${protobuf_libprotobuf_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${protobuf_libprotobuf_LINK_LIBS_RELEASE}
                ${protobuf_libprotobuf_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${protobuf_libprotobuf_LINK_LIBS_RELWITHDEBINFO}
                ${protobuf_libprotobuf_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${protobuf_libprotobuf_LINK_LIBS_MINSIZEREL}
                ${protobuf_libprotobuf_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET protobuf::libprotobuf PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${protobuf_libprotobuf_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${protobuf_libprotobuf_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${protobuf_libprotobuf_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${protobuf_libprotobuf_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET protobuf::libprotobuf PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${protobuf_libprotobuf_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${protobuf_libprotobuf_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${protobuf_libprotobuf_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${protobuf_libprotobuf_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET protobuf::libprotobuf PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${protobuf_libprotobuf_COMPILE_OPTIONS_C_DEBUG}
                 ${protobuf_libprotobuf_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${protobuf_libprotobuf_COMPILE_OPTIONS_C_RELEASE}
                 ${protobuf_libprotobuf_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${protobuf_libprotobuf_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${protobuf_libprotobuf_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${protobuf_libprotobuf_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${protobuf_libprotobuf_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(protobuf_libprotobuf_TARGET_PROPERTIES TRUE)

########## COMPONENT libprotoc TARGET PROPERTIES ######################################

set_property(TARGET protobuf::libprotoc PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${protobuf_libprotoc_LINK_LIBS_DEBUG}
                ${protobuf_libprotoc_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${protobuf_libprotoc_LINK_LIBS_RELEASE}
                ${protobuf_libprotoc_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${protobuf_libprotoc_LINK_LIBS_RELWITHDEBINFO}
                ${protobuf_libprotoc_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${protobuf_libprotoc_LINK_LIBS_MINSIZEREL}
                ${protobuf_libprotoc_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET protobuf::libprotoc PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${protobuf_libprotoc_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${protobuf_libprotoc_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${protobuf_libprotoc_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${protobuf_libprotoc_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET protobuf::libprotoc PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${protobuf_libprotoc_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${protobuf_libprotoc_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${protobuf_libprotoc_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${protobuf_libprotoc_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET protobuf::libprotoc PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${protobuf_libprotoc_COMPILE_OPTIONS_C_DEBUG}
                 ${protobuf_libprotoc_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${protobuf_libprotoc_COMPILE_OPTIONS_C_RELEASE}
                 ${protobuf_libprotoc_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${protobuf_libprotoc_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${protobuf_libprotoc_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${protobuf_libprotoc_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${protobuf_libprotoc_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(protobuf_libprotoc_TARGET_PROPERTIES TRUE)

########## GLOBAL TARGET PROPERTIES #########################################################

if(NOT protobuf_protobuf_TARGET_PROPERTIES)
    set_property(TARGET protobuf::protobuf APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                 $<$<CONFIG:Debug>:${protobuf_COMPONENTS_DEBUG}>
                 $<$<CONFIG:Release>:${protobuf_COMPONENTS_RELEASE}>
                 $<$<CONFIG:RelWithDebInfo>:${protobuf_COMPONENTS_RELWITHDEBINFO}>
                 $<$<CONFIG:MinSizeRel>:${protobuf_COMPONENTS_MINSIZEREL}>)
endif()

########## BUILD MODULES ####################################################################
#############################################################################################

########## COMPONENT libprotobuf BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${protobuf_libprotobuf_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${protobuf_libprotobuf_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${protobuf_libprotobuf_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${protobuf_libprotobuf_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT libprotoc BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${protobuf_libprotoc_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${protobuf_libprotoc_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${protobuf_libprotoc_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${protobuf_libprotoc_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()