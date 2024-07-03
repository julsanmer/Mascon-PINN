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

include(${CMAKE_CURRENT_LIST_DIR}/Microsoft.GSLTargets.cmake)

########## FIND PACKAGE DEPENDENCY ##########################################################
#############################################################################################

include(CMakeFindDependencyMacro)

########## TARGETS PROPERTIES ###############################################################
#############################################################################################

########## COMPONENT GSL TARGET PROPERTIES ######################################

set_property(TARGET Microsoft.GSL::GSL PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${Microsoft.GSL_GSL_LINK_LIBS_DEBUG}
                ${Microsoft.GSL_GSL_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${Microsoft.GSL_GSL_LINK_LIBS_RELEASE}
                ${Microsoft.GSL_GSL_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Microsoft.GSL_GSL_LINK_LIBS_RELWITHDEBINFO}
                ${Microsoft.GSL_GSL_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Microsoft.GSL_GSL_LINK_LIBS_MINSIZEREL}
                ${Microsoft.GSL_GSL_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET Microsoft.GSL::GSL PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${Microsoft.GSL_GSL_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${Microsoft.GSL_GSL_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Microsoft.GSL_GSL_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Microsoft.GSL_GSL_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET Microsoft.GSL::GSL PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${Microsoft.GSL_GSL_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${Microsoft.GSL_GSL_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Microsoft.GSL_GSL_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Microsoft.GSL_GSL_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET Microsoft.GSL::GSL PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${Microsoft.GSL_GSL_COMPILE_OPTIONS_C_DEBUG}
                 ${Microsoft.GSL_GSL_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${Microsoft.GSL_GSL_COMPILE_OPTIONS_C_RELEASE}
                 ${Microsoft.GSL_GSL_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${Microsoft.GSL_GSL_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${Microsoft.GSL_GSL_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${Microsoft.GSL_GSL_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${Microsoft.GSL_GSL_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(Microsoft.GSL_GSL_TARGET_PROPERTIES TRUE)

########## GLOBAL TARGET PROPERTIES #########################################################

if(NOT Microsoft.GSL_Microsoft.GSL_TARGET_PROPERTIES)
    set_property(TARGET Microsoft.GSL::Microsoft.GSL APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                 $<$<CONFIG:Debug>:${Microsoft.GSL_COMPONENTS_DEBUG}>
                 $<$<CONFIG:Release>:${Microsoft.GSL_COMPONENTS_RELEASE}>
                 $<$<CONFIG:RelWithDebInfo>:${Microsoft.GSL_COMPONENTS_RELWITHDEBINFO}>
                 $<$<CONFIG:MinSizeRel>:${Microsoft.GSL_COMPONENTS_MINSIZEREL}>)
endif()

########## BUILD MODULES ####################################################################
#############################################################################################

########## COMPONENT GSL BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${Microsoft.GSL_GSL_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Microsoft.GSL_GSL_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Microsoft.GSL_GSL_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Microsoft.GSL_GSL_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()