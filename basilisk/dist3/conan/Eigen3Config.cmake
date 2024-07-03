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

include(${CMAKE_CURRENT_LIST_DIR}/Eigen3Targets.cmake)

########## FIND PACKAGE DEPENDENCY ##########################################################
#############################################################################################

include(CMakeFindDependencyMacro)

########## TARGETS PROPERTIES ###############################################################
#############################################################################################

########## COMPONENT Eigen TARGET PROPERTIES ######################################

set_property(TARGET Eigen3::Eigen PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${Eigen3_Eigen_LINK_LIBS_DEBUG}
                ${Eigen3_Eigen_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${Eigen3_Eigen_LINK_LIBS_RELEASE}
                ${Eigen3_Eigen_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Eigen3_Eigen_LINK_LIBS_RELWITHDEBINFO}
                ${Eigen3_Eigen_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Eigen3_Eigen_LINK_LIBS_MINSIZEREL}
                ${Eigen3_Eigen_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET Eigen3::Eigen PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${Eigen3_Eigen_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${Eigen3_Eigen_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Eigen3_Eigen_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Eigen3_Eigen_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET Eigen3::Eigen PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${Eigen3_Eigen_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${Eigen3_Eigen_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Eigen3_Eigen_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Eigen3_Eigen_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET Eigen3::Eigen PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${Eigen3_Eigen_COMPILE_OPTIONS_C_DEBUG}
                 ${Eigen3_Eigen_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${Eigen3_Eigen_COMPILE_OPTIONS_C_RELEASE}
                 ${Eigen3_Eigen_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${Eigen3_Eigen_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${Eigen3_Eigen_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${Eigen3_Eigen_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${Eigen3_Eigen_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(Eigen3_Eigen_TARGET_PROPERTIES TRUE)

########## GLOBAL TARGET PROPERTIES #########################################################

if(NOT Eigen3_Eigen3_TARGET_PROPERTIES)
    set_property(TARGET Eigen3::Eigen3 APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                 $<$<CONFIG:Debug>:${Eigen3_COMPONENTS_DEBUG}>
                 $<$<CONFIG:Release>:${Eigen3_COMPONENTS_RELEASE}>
                 $<$<CONFIG:RelWithDebInfo>:${Eigen3_COMPONENTS_RELWITHDEBINFO}>
                 $<$<CONFIG:MinSizeRel>:${Eigen3_COMPONENTS_MINSIZEREL}>)
endif()

########## BUILD MODULES ####################################################################
#############################################################################################

########## COMPONENT Eigen BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${Eigen3_Eigen_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Eigen3_Eigen_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Eigen3_Eigen_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Eigen3_Eigen_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()