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

include(${CMAKE_CURRENT_LIST_DIR}/cpuinfoTargets.cmake)

########## FIND PACKAGE DEPENDENCY ##########################################################
#############################################################################################

include(CMakeFindDependencyMacro)

########## TARGETS PROPERTIES ###############################################################
#############################################################################################

########## COMPONENT clog TARGET PROPERTIES ######################################

set_property(TARGET cpuinfo::clog PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${cpuinfo_clog_LINK_LIBS_DEBUG}
                ${cpuinfo_clog_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${cpuinfo_clog_LINK_LIBS_RELEASE}
                ${cpuinfo_clog_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${cpuinfo_clog_LINK_LIBS_RELWITHDEBINFO}
                ${cpuinfo_clog_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${cpuinfo_clog_LINK_LIBS_MINSIZEREL}
                ${cpuinfo_clog_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET cpuinfo::clog PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${cpuinfo_clog_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${cpuinfo_clog_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${cpuinfo_clog_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${cpuinfo_clog_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET cpuinfo::clog PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${cpuinfo_clog_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${cpuinfo_clog_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${cpuinfo_clog_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${cpuinfo_clog_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET cpuinfo::clog PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${cpuinfo_clog_COMPILE_OPTIONS_C_DEBUG}
                 ${cpuinfo_clog_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${cpuinfo_clog_COMPILE_OPTIONS_C_RELEASE}
                 ${cpuinfo_clog_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${cpuinfo_clog_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${cpuinfo_clog_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${cpuinfo_clog_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${cpuinfo_clog_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(cpuinfo_clog_TARGET_PROPERTIES TRUE)

########## COMPONENT cpuinfo TARGET PROPERTIES ######################################

set_property(TARGET cpuinfo::cpuinfo PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${cpuinfo_cpuinfo_LINK_LIBS_DEBUG}
                ${cpuinfo_cpuinfo_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${cpuinfo_cpuinfo_LINK_LIBS_RELEASE}
                ${cpuinfo_cpuinfo_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${cpuinfo_cpuinfo_LINK_LIBS_RELWITHDEBINFO}
                ${cpuinfo_cpuinfo_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${cpuinfo_cpuinfo_LINK_LIBS_MINSIZEREL}
                ${cpuinfo_cpuinfo_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET cpuinfo::cpuinfo PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${cpuinfo_cpuinfo_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${cpuinfo_cpuinfo_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${cpuinfo_cpuinfo_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${cpuinfo_cpuinfo_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET cpuinfo::cpuinfo PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${cpuinfo_cpuinfo_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${cpuinfo_cpuinfo_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${cpuinfo_cpuinfo_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${cpuinfo_cpuinfo_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET cpuinfo::cpuinfo PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${cpuinfo_cpuinfo_COMPILE_OPTIONS_C_DEBUG}
                 ${cpuinfo_cpuinfo_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${cpuinfo_cpuinfo_COMPILE_OPTIONS_C_RELEASE}
                 ${cpuinfo_cpuinfo_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${cpuinfo_cpuinfo_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${cpuinfo_cpuinfo_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${cpuinfo_cpuinfo_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${cpuinfo_cpuinfo_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(cpuinfo_cpuinfo_TARGET_PROPERTIES TRUE)

########## GLOBAL TARGET PROPERTIES #########################################################

if(NOT cpuinfo_cpuinfo_TARGET_PROPERTIES)
    set_property(TARGET cpuinfo::cpuinfo APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                 $<$<CONFIG:Debug>:${cpuinfo_COMPONENTS_DEBUG}>
                 $<$<CONFIG:Release>:${cpuinfo_COMPONENTS_RELEASE}>
                 $<$<CONFIG:RelWithDebInfo>:${cpuinfo_COMPONENTS_RELWITHDEBINFO}>
                 $<$<CONFIG:MinSizeRel>:${cpuinfo_COMPONENTS_MINSIZEREL}>)
endif()

########## BUILD MODULES ####################################################################
#############################################################################################

########## COMPONENT clog BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${cpuinfo_clog_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${cpuinfo_clog_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${cpuinfo_clog_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${cpuinfo_clog_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT cpuinfo BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${cpuinfo_cpuinfo_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${cpuinfo_cpuinfo_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${cpuinfo_cpuinfo_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${cpuinfo_cpuinfo_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()