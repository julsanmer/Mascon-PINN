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

include(${CMAKE_CURRENT_LIST_DIR}/flatbuffersTargets.cmake)

########## FIND PACKAGE DEPENDENCY ##########################################################
#############################################################################################

include(CMakeFindDependencyMacro)

########## TARGETS PROPERTIES ###############################################################
#############################################################################################

########## COMPONENT flatbuffers TARGET PROPERTIES ######################################

set_property(TARGET flatbuffers::flatbuffers PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${flatbuffers_flatbuffers_LINK_LIBS_DEBUG}
                ${flatbuffers_flatbuffers_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${flatbuffers_flatbuffers_LINK_LIBS_RELEASE}
                ${flatbuffers_flatbuffers_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${flatbuffers_flatbuffers_LINK_LIBS_RELWITHDEBINFO}
                ${flatbuffers_flatbuffers_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${flatbuffers_flatbuffers_LINK_LIBS_MINSIZEREL}
                ${flatbuffers_flatbuffers_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET flatbuffers::flatbuffers PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${flatbuffers_flatbuffers_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${flatbuffers_flatbuffers_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${flatbuffers_flatbuffers_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${flatbuffers_flatbuffers_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET flatbuffers::flatbuffers PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${flatbuffers_flatbuffers_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${flatbuffers_flatbuffers_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${flatbuffers_flatbuffers_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${flatbuffers_flatbuffers_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET flatbuffers::flatbuffers PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${flatbuffers_flatbuffers_COMPILE_OPTIONS_C_DEBUG}
                 ${flatbuffers_flatbuffers_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${flatbuffers_flatbuffers_COMPILE_OPTIONS_C_RELEASE}
                 ${flatbuffers_flatbuffers_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${flatbuffers_flatbuffers_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${flatbuffers_flatbuffers_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${flatbuffers_flatbuffers_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${flatbuffers_flatbuffers_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(flatbuffers_flatbuffers_TARGET_PROPERTIES TRUE)

########## GLOBAL TARGET PROPERTIES #########################################################

if(NOT flatbuffers_flatbuffers_TARGET_PROPERTIES)
    set_property(TARGET flatbuffers::flatbuffers APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                 $<$<CONFIG:Debug>:${flatbuffers_COMPONENTS_DEBUG}>
                 $<$<CONFIG:Release>:${flatbuffers_COMPONENTS_RELEASE}>
                 $<$<CONFIG:RelWithDebInfo>:${flatbuffers_COMPONENTS_RELWITHDEBINFO}>
                 $<$<CONFIG:MinSizeRel>:${flatbuffers_COMPONENTS_MINSIZEREL}>)
endif()

########## BUILD MODULES ####################################################################
#############################################################################################

########## COMPONENT flatbuffers BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${flatbuffers_flatbuffers_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${flatbuffers_flatbuffers_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${flatbuffers_flatbuffers_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${flatbuffers_flatbuffers_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()