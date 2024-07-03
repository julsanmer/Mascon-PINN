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

include(${CMAKE_CURRENT_LIST_DIR}/nsyncTargets.cmake)

########## FIND PACKAGE DEPENDENCY ##########################################################
#############################################################################################

include(CMakeFindDependencyMacro)

########## TARGETS PROPERTIES ###############################################################
#############################################################################################

########## COMPONENT nsync_cpp TARGET PROPERTIES ######################################

set_property(TARGET nsync::nsync_cpp PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${nsync_nsync_cpp_LINK_LIBS_DEBUG}
                ${nsync_nsync_cpp_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${nsync_nsync_cpp_LINK_LIBS_RELEASE}
                ${nsync_nsync_cpp_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${nsync_nsync_cpp_LINK_LIBS_RELWITHDEBINFO}
                ${nsync_nsync_cpp_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${nsync_nsync_cpp_LINK_LIBS_MINSIZEREL}
                ${nsync_nsync_cpp_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET nsync::nsync_cpp PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${nsync_nsync_cpp_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${nsync_nsync_cpp_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${nsync_nsync_cpp_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${nsync_nsync_cpp_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET nsync::nsync_cpp PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${nsync_nsync_cpp_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${nsync_nsync_cpp_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${nsync_nsync_cpp_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${nsync_nsync_cpp_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET nsync::nsync_cpp PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${nsync_nsync_cpp_COMPILE_OPTIONS_C_DEBUG}
                 ${nsync_nsync_cpp_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${nsync_nsync_cpp_COMPILE_OPTIONS_C_RELEASE}
                 ${nsync_nsync_cpp_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${nsync_nsync_cpp_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${nsync_nsync_cpp_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${nsync_nsync_cpp_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${nsync_nsync_cpp_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(nsync_nsync_cpp_TARGET_PROPERTIES TRUE)

########## COMPONENT nsync_c TARGET PROPERTIES ######################################

set_property(TARGET nsync::nsync_c PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${nsync_nsync_c_LINK_LIBS_DEBUG}
                ${nsync_nsync_c_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${nsync_nsync_c_LINK_LIBS_RELEASE}
                ${nsync_nsync_c_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${nsync_nsync_c_LINK_LIBS_RELWITHDEBINFO}
                ${nsync_nsync_c_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${nsync_nsync_c_LINK_LIBS_MINSIZEREL}
                ${nsync_nsync_c_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET nsync::nsync_c PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${nsync_nsync_c_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${nsync_nsync_c_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${nsync_nsync_c_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${nsync_nsync_c_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET nsync::nsync_c PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${nsync_nsync_c_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${nsync_nsync_c_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${nsync_nsync_c_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${nsync_nsync_c_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET nsync::nsync_c PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${nsync_nsync_c_COMPILE_OPTIONS_C_DEBUG}
                 ${nsync_nsync_c_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${nsync_nsync_c_COMPILE_OPTIONS_C_RELEASE}
                 ${nsync_nsync_c_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${nsync_nsync_c_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${nsync_nsync_c_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${nsync_nsync_c_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${nsync_nsync_c_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(nsync_nsync_c_TARGET_PROPERTIES TRUE)

########## GLOBAL TARGET PROPERTIES #########################################################

if(NOT nsync_nsync_TARGET_PROPERTIES)
    set_property(TARGET nsync::nsync APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                 $<$<CONFIG:Debug>:${nsync_COMPONENTS_DEBUG}>
                 $<$<CONFIG:Release>:${nsync_COMPONENTS_RELEASE}>
                 $<$<CONFIG:RelWithDebInfo>:${nsync_COMPONENTS_RELWITHDEBINFO}>
                 $<$<CONFIG:MinSizeRel>:${nsync_COMPONENTS_MINSIZEREL}>)
endif()

########## BUILD MODULES ####################################################################
#############################################################################################

########## COMPONENT nsync_cpp BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${nsync_nsync_cpp_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${nsync_nsync_cpp_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${nsync_nsync_cpp_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${nsync_nsync_cpp_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT nsync_c BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${nsync_nsync_c_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${nsync_nsync_c_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${nsync_nsync_c_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${nsync_nsync_c_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()