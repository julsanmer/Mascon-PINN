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

include(${CMAKE_CURRENT_LIST_DIR}/dateTargets.cmake)

########## FIND PACKAGE DEPENDENCY ##########################################################
#############################################################################################

include(CMakeFindDependencyMacro)

if(NOT CURL_FOUND)
    if(${CMAKE_VERSION} VERSION_LESS "3.9.0")
        find_package(CURL REQUIRED NO_MODULE)
    else()
        find_dependency(CURL REQUIRED NO_MODULE)
    endif()
else()
    message(STATUS "Dependency CURL already found")
endif()

########## TARGETS PROPERTIES ###############################################################
#############################################################################################

########## COMPONENT date-tz TARGET PROPERTIES ######################################

set_property(TARGET date::date-tz PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${date_date-tz_LINK_LIBS_DEBUG}
                ${date_date-tz_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${date_date-tz_LINK_LIBS_RELEASE}
                ${date_date-tz_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${date_date-tz_LINK_LIBS_RELWITHDEBINFO}
                ${date_date-tz_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${date_date-tz_LINK_LIBS_MINSIZEREL}
                ${date_date-tz_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET date::date-tz PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${date_date-tz_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${date_date-tz_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${date_date-tz_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${date_date-tz_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET date::date-tz PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${date_date-tz_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${date_date-tz_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${date_date-tz_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${date_date-tz_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET date::date-tz PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${date_date-tz_COMPILE_OPTIONS_C_DEBUG}
                 ${date_date-tz_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${date_date-tz_COMPILE_OPTIONS_C_RELEASE}
                 ${date_date-tz_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${date_date-tz_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${date_date-tz_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${date_date-tz_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${date_date-tz_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(date_date-tz_TARGET_PROPERTIES TRUE)

########## GLOBAL TARGET PROPERTIES #########################################################

if(NOT date_date_TARGET_PROPERTIES)
    set_property(TARGET date::date APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                 $<$<CONFIG:Debug>:${date_COMPONENTS_DEBUG}>
                 $<$<CONFIG:Release>:${date_COMPONENTS_RELEASE}>
                 $<$<CONFIG:RelWithDebInfo>:${date_COMPONENTS_RELWITHDEBINFO}>
                 $<$<CONFIG:MinSizeRel>:${date_COMPONENTS_MINSIZEREL}>)
endif()

########## BUILD MODULES ####################################################################
#############################################################################################

########## COMPONENT date-tz BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${date_date-tz_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${date_date-tz_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${date_date-tz_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${date_date-tz_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()