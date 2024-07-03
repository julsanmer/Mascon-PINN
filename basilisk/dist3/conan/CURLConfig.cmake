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

include(${CMAKE_CURRENT_LIST_DIR}/CURLTargets.cmake)

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

########## COMPONENT libcurl TARGET PROPERTIES ######################################

set_property(TARGET CURL::libcurl PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${CURL_libcurl_LINK_LIBS_DEBUG}
                ${CURL_libcurl_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${CURL_libcurl_LINK_LIBS_RELEASE}
                ${CURL_libcurl_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${CURL_libcurl_LINK_LIBS_RELWITHDEBINFO}
                ${CURL_libcurl_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${CURL_libcurl_LINK_LIBS_MINSIZEREL}
                ${CURL_libcurl_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET CURL::libcurl PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${CURL_libcurl_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${CURL_libcurl_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${CURL_libcurl_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${CURL_libcurl_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET CURL::libcurl PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${CURL_libcurl_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${CURL_libcurl_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${CURL_libcurl_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${CURL_libcurl_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET CURL::libcurl PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${CURL_libcurl_COMPILE_OPTIONS_C_DEBUG}
                 ${CURL_libcurl_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${CURL_libcurl_COMPILE_OPTIONS_C_RELEASE}
                 ${CURL_libcurl_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${CURL_libcurl_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${CURL_libcurl_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${CURL_libcurl_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${CURL_libcurl_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(CURL_libcurl_TARGET_PROPERTIES TRUE)

########## GLOBAL TARGET PROPERTIES #########################################################

if(NOT CURL_CURL_TARGET_PROPERTIES)
    set_property(TARGET CURL::CURL APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                 $<$<CONFIG:Debug>:${CURL_COMPONENTS_DEBUG}>
                 $<$<CONFIG:Release>:${CURL_COMPONENTS_RELEASE}>
                 $<$<CONFIG:RelWithDebInfo>:${CURL_COMPONENTS_RELWITHDEBINFO}>
                 $<$<CONFIG:MinSizeRel>:${CURL_COMPONENTS_MINSIZEREL}>)
endif()

########## BUILD MODULES ####################################################################
#############################################################################################

########## COMPONENT libcurl BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${CURL_libcurl_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${CURL_libcurl_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${CURL_libcurl_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${CURL_libcurl_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()