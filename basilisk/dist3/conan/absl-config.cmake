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

include(${CMAKE_CURRENT_LIST_DIR}/abslTargets.cmake)

########## FIND PACKAGE DEPENDENCY ##########################################################
#############################################################################################

include(CMakeFindDependencyMacro)

########## TARGETS PROPERTIES ###############################################################
#############################################################################################

########## COMPONENT config TARGET PROPERTIES ######################################

set_property(TARGET absl::config PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_config_LINK_LIBS_DEBUG}
                ${absl_config_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_config_LINK_LIBS_RELEASE}
                ${absl_config_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_config_LINK_LIBS_RELWITHDEBINFO}
                ${absl_config_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_config_LINK_LIBS_MINSIZEREL}
                ${absl_config_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::config PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_config_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_config_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_config_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_config_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::config PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_config_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_config_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_config_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_config_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::config PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_config_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_config_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_config_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_config_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_config_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_config_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_config_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_config_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_config_TARGET_PROPERTIES TRUE)

########## COMPONENT type_traits TARGET PROPERTIES ######################################

set_property(TARGET absl::type_traits PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_type_traits_LINK_LIBS_DEBUG}
                ${absl_type_traits_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_type_traits_LINK_LIBS_RELEASE}
                ${absl_type_traits_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_type_traits_LINK_LIBS_RELWITHDEBINFO}
                ${absl_type_traits_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_type_traits_LINK_LIBS_MINSIZEREL}
                ${absl_type_traits_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::type_traits PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_type_traits_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_type_traits_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_type_traits_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_type_traits_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::type_traits PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_type_traits_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_type_traits_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_type_traits_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_type_traits_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::type_traits PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_type_traits_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_type_traits_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_type_traits_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_type_traits_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_type_traits_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_type_traits_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_type_traits_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_type_traits_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_type_traits_TARGET_PROPERTIES TRUE)

########## COMPONENT base_internal TARGET PROPERTIES ######################################

set_property(TARGET absl::base_internal PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_base_internal_LINK_LIBS_DEBUG}
                ${absl_base_internal_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_base_internal_LINK_LIBS_RELEASE}
                ${absl_base_internal_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_base_internal_LINK_LIBS_RELWITHDEBINFO}
                ${absl_base_internal_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_base_internal_LINK_LIBS_MINSIZEREL}
                ${absl_base_internal_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::base_internal PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_base_internal_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_base_internal_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_base_internal_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_base_internal_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::base_internal PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_base_internal_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_base_internal_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_base_internal_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_base_internal_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::base_internal PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_base_internal_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_base_internal_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_base_internal_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_base_internal_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_base_internal_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_base_internal_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_base_internal_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_base_internal_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_base_internal_TARGET_PROPERTIES TRUE)

########## COMPONENT utility TARGET PROPERTIES ######################################

set_property(TARGET absl::utility PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_utility_LINK_LIBS_DEBUG}
                ${absl_utility_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_utility_LINK_LIBS_RELEASE}
                ${absl_utility_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_utility_LINK_LIBS_RELWITHDEBINFO}
                ${absl_utility_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_utility_LINK_LIBS_MINSIZEREL}
                ${absl_utility_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::utility PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_utility_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_utility_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_utility_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_utility_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::utility PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_utility_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_utility_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_utility_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_utility_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::utility PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_utility_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_utility_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_utility_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_utility_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_utility_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_utility_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_utility_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_utility_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_utility_TARGET_PROPERTIES TRUE)

########## COMPONENT core_headers TARGET PROPERTIES ######################################

set_property(TARGET absl::core_headers PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_core_headers_LINK_LIBS_DEBUG}
                ${absl_core_headers_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_core_headers_LINK_LIBS_RELEASE}
                ${absl_core_headers_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_core_headers_LINK_LIBS_RELWITHDEBINFO}
                ${absl_core_headers_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_core_headers_LINK_LIBS_MINSIZEREL}
                ${absl_core_headers_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::core_headers PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_core_headers_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_core_headers_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_core_headers_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_core_headers_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::core_headers PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_core_headers_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_core_headers_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_core_headers_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_core_headers_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::core_headers PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_core_headers_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_core_headers_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_core_headers_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_core_headers_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_core_headers_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_core_headers_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_core_headers_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_core_headers_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_core_headers_TARGET_PROPERTIES TRUE)

########## COMPONENT compare TARGET PROPERTIES ######################################

set_property(TARGET absl::compare PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_compare_LINK_LIBS_DEBUG}
                ${absl_compare_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_compare_LINK_LIBS_RELEASE}
                ${absl_compare_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_compare_LINK_LIBS_RELWITHDEBINFO}
                ${absl_compare_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_compare_LINK_LIBS_MINSIZEREL}
                ${absl_compare_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::compare PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_compare_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_compare_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_compare_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_compare_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::compare PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_compare_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_compare_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_compare_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_compare_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::compare PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_compare_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_compare_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_compare_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_compare_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_compare_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_compare_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_compare_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_compare_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_compare_TARGET_PROPERTIES TRUE)

########## COMPONENT log_severity TARGET PROPERTIES ######################################

set_property(TARGET absl::log_severity PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_log_severity_LINK_LIBS_DEBUG}
                ${absl_log_severity_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_log_severity_LINK_LIBS_RELEASE}
                ${absl_log_severity_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_log_severity_LINK_LIBS_RELWITHDEBINFO}
                ${absl_log_severity_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_log_severity_LINK_LIBS_MINSIZEREL}
                ${absl_log_severity_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::log_severity PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_log_severity_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_log_severity_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_log_severity_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_log_severity_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::log_severity PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_log_severity_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_log_severity_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_log_severity_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_log_severity_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::log_severity PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_log_severity_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_log_severity_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_log_severity_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_log_severity_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_log_severity_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_log_severity_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_log_severity_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_log_severity_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_log_severity_TARGET_PROPERTIES TRUE)

########## COMPONENT errno_saver TARGET PROPERTIES ######################################

set_property(TARGET absl::errno_saver PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_errno_saver_LINK_LIBS_DEBUG}
                ${absl_errno_saver_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_errno_saver_LINK_LIBS_RELEASE}
                ${absl_errno_saver_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_errno_saver_LINK_LIBS_RELWITHDEBINFO}
                ${absl_errno_saver_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_errno_saver_LINK_LIBS_MINSIZEREL}
                ${absl_errno_saver_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::errno_saver PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_errno_saver_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_errno_saver_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_errno_saver_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_errno_saver_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::errno_saver PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_errno_saver_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_errno_saver_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_errno_saver_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_errno_saver_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::errno_saver PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_errno_saver_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_errno_saver_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_errno_saver_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_errno_saver_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_errno_saver_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_errno_saver_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_errno_saver_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_errno_saver_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_errno_saver_TARGET_PROPERTIES TRUE)

########## COMPONENT atomic_hook TARGET PROPERTIES ######################################

set_property(TARGET absl::atomic_hook PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_atomic_hook_LINK_LIBS_DEBUG}
                ${absl_atomic_hook_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_atomic_hook_LINK_LIBS_RELEASE}
                ${absl_atomic_hook_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_atomic_hook_LINK_LIBS_RELWITHDEBINFO}
                ${absl_atomic_hook_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_atomic_hook_LINK_LIBS_MINSIZEREL}
                ${absl_atomic_hook_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::atomic_hook PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_atomic_hook_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_atomic_hook_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_atomic_hook_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_atomic_hook_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::atomic_hook PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_atomic_hook_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_atomic_hook_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_atomic_hook_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_atomic_hook_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::atomic_hook PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_atomic_hook_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_atomic_hook_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_atomic_hook_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_atomic_hook_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_atomic_hook_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_atomic_hook_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_atomic_hook_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_atomic_hook_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_atomic_hook_TARGET_PROPERTIES TRUE)

########## COMPONENT raw_logging_internal TARGET PROPERTIES ######################################

set_property(TARGET absl::raw_logging_internal PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_raw_logging_internal_LINK_LIBS_DEBUG}
                ${absl_raw_logging_internal_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_raw_logging_internal_LINK_LIBS_RELEASE}
                ${absl_raw_logging_internal_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_raw_logging_internal_LINK_LIBS_RELWITHDEBINFO}
                ${absl_raw_logging_internal_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_raw_logging_internal_LINK_LIBS_MINSIZEREL}
                ${absl_raw_logging_internal_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::raw_logging_internal PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_raw_logging_internal_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_raw_logging_internal_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_raw_logging_internal_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_raw_logging_internal_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::raw_logging_internal PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_raw_logging_internal_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_raw_logging_internal_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_raw_logging_internal_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_raw_logging_internal_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::raw_logging_internal PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_raw_logging_internal_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_raw_logging_internal_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_raw_logging_internal_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_raw_logging_internal_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_raw_logging_internal_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_raw_logging_internal_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_raw_logging_internal_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_raw_logging_internal_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_raw_logging_internal_TARGET_PROPERTIES TRUE)

########## COMPONENT bad_variant_access TARGET PROPERTIES ######################################

set_property(TARGET absl::bad_variant_access PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_bad_variant_access_LINK_LIBS_DEBUG}
                ${absl_bad_variant_access_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_bad_variant_access_LINK_LIBS_RELEASE}
                ${absl_bad_variant_access_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_bad_variant_access_LINK_LIBS_RELWITHDEBINFO}
                ${absl_bad_variant_access_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_bad_variant_access_LINK_LIBS_MINSIZEREL}
                ${absl_bad_variant_access_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::bad_variant_access PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_bad_variant_access_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_bad_variant_access_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_bad_variant_access_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_bad_variant_access_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::bad_variant_access PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_bad_variant_access_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_bad_variant_access_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_bad_variant_access_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_bad_variant_access_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::bad_variant_access PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_bad_variant_access_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_bad_variant_access_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_bad_variant_access_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_bad_variant_access_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_bad_variant_access_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_bad_variant_access_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_bad_variant_access_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_bad_variant_access_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_bad_variant_access_TARGET_PROPERTIES TRUE)

########## COMPONENT variant TARGET PROPERTIES ######################################

set_property(TARGET absl::variant PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_variant_LINK_LIBS_DEBUG}
                ${absl_variant_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_variant_LINK_LIBS_RELEASE}
                ${absl_variant_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_variant_LINK_LIBS_RELWITHDEBINFO}
                ${absl_variant_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_variant_LINK_LIBS_MINSIZEREL}
                ${absl_variant_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::variant PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_variant_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_variant_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_variant_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_variant_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::variant PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_variant_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_variant_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_variant_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_variant_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::variant PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_variant_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_variant_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_variant_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_variant_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_variant_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_variant_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_variant_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_variant_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_variant_TARGET_PROPERTIES TRUE)

########## COMPONENT bad_optional_access TARGET PROPERTIES ######################################

set_property(TARGET absl::bad_optional_access PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_bad_optional_access_LINK_LIBS_DEBUG}
                ${absl_bad_optional_access_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_bad_optional_access_LINK_LIBS_RELEASE}
                ${absl_bad_optional_access_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_bad_optional_access_LINK_LIBS_RELWITHDEBINFO}
                ${absl_bad_optional_access_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_bad_optional_access_LINK_LIBS_MINSIZEREL}
                ${absl_bad_optional_access_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::bad_optional_access PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_bad_optional_access_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_bad_optional_access_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_bad_optional_access_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_bad_optional_access_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::bad_optional_access PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_bad_optional_access_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_bad_optional_access_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_bad_optional_access_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_bad_optional_access_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::bad_optional_access PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_bad_optional_access_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_bad_optional_access_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_bad_optional_access_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_bad_optional_access_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_bad_optional_access_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_bad_optional_access_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_bad_optional_access_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_bad_optional_access_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_bad_optional_access_TARGET_PROPERTIES TRUE)

########## COMPONENT meta TARGET PROPERTIES ######################################

set_property(TARGET absl::meta PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_meta_LINK_LIBS_DEBUG}
                ${absl_meta_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_meta_LINK_LIBS_RELEASE}
                ${absl_meta_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_meta_LINK_LIBS_RELWITHDEBINFO}
                ${absl_meta_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_meta_LINK_LIBS_MINSIZEREL}
                ${absl_meta_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::meta PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_meta_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_meta_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_meta_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_meta_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::meta PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_meta_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_meta_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_meta_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_meta_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::meta PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_meta_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_meta_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_meta_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_meta_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_meta_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_meta_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_meta_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_meta_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_meta_TARGET_PROPERTIES TRUE)

########## COMPONENT memory TARGET PROPERTIES ######################################

set_property(TARGET absl::memory PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_memory_LINK_LIBS_DEBUG}
                ${absl_memory_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_memory_LINK_LIBS_RELEASE}
                ${absl_memory_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_memory_LINK_LIBS_RELWITHDEBINFO}
                ${absl_memory_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_memory_LINK_LIBS_MINSIZEREL}
                ${absl_memory_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::memory PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_memory_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_memory_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_memory_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_memory_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::memory PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_memory_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_memory_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_memory_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_memory_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::memory PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_memory_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_memory_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_memory_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_memory_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_memory_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_memory_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_memory_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_memory_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_memory_TARGET_PROPERTIES TRUE)

########## COMPONENT optional TARGET PROPERTIES ######################################

set_property(TARGET absl::optional PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_optional_LINK_LIBS_DEBUG}
                ${absl_optional_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_optional_LINK_LIBS_RELEASE}
                ${absl_optional_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_optional_LINK_LIBS_RELWITHDEBINFO}
                ${absl_optional_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_optional_LINK_LIBS_MINSIZEREL}
                ${absl_optional_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::optional PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_optional_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_optional_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_optional_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_optional_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::optional PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_optional_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_optional_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_optional_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_optional_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::optional PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_optional_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_optional_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_optional_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_optional_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_optional_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_optional_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_optional_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_optional_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_optional_TARGET_PROPERTIES TRUE)

########## COMPONENT algorithm TARGET PROPERTIES ######################################

set_property(TARGET absl::algorithm PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_algorithm_LINK_LIBS_DEBUG}
                ${absl_algorithm_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_algorithm_LINK_LIBS_RELEASE}
                ${absl_algorithm_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_algorithm_LINK_LIBS_RELWITHDEBINFO}
                ${absl_algorithm_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_algorithm_LINK_LIBS_MINSIZEREL}
                ${absl_algorithm_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::algorithm PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_algorithm_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_algorithm_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_algorithm_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_algorithm_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::algorithm PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_algorithm_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_algorithm_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_algorithm_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_algorithm_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::algorithm PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_algorithm_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_algorithm_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_algorithm_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_algorithm_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_algorithm_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_algorithm_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_algorithm_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_algorithm_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_algorithm_TARGET_PROPERTIES TRUE)

########## COMPONENT throw_delegate TARGET PROPERTIES ######################################

set_property(TARGET absl::throw_delegate PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_throw_delegate_LINK_LIBS_DEBUG}
                ${absl_throw_delegate_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_throw_delegate_LINK_LIBS_RELEASE}
                ${absl_throw_delegate_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_throw_delegate_LINK_LIBS_RELWITHDEBINFO}
                ${absl_throw_delegate_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_throw_delegate_LINK_LIBS_MINSIZEREL}
                ${absl_throw_delegate_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::throw_delegate PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_throw_delegate_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_throw_delegate_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_throw_delegate_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_throw_delegate_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::throw_delegate PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_throw_delegate_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_throw_delegate_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_throw_delegate_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_throw_delegate_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::throw_delegate PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_throw_delegate_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_throw_delegate_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_throw_delegate_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_throw_delegate_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_throw_delegate_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_throw_delegate_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_throw_delegate_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_throw_delegate_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_throw_delegate_TARGET_PROPERTIES TRUE)

########## COMPONENT span TARGET PROPERTIES ######################################

set_property(TARGET absl::span PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_span_LINK_LIBS_DEBUG}
                ${absl_span_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_span_LINK_LIBS_RELEASE}
                ${absl_span_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_span_LINK_LIBS_RELWITHDEBINFO}
                ${absl_span_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_span_LINK_LIBS_MINSIZEREL}
                ${absl_span_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::span PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_span_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_span_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_span_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_span_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::span PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_span_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_span_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_span_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_span_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::span PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_span_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_span_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_span_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_span_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_span_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_span_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_span_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_span_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_span_TARGET_PROPERTIES TRUE)

########## COMPONENT bad_any_cast_impl TARGET PROPERTIES ######################################

set_property(TARGET absl::bad_any_cast_impl PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_bad_any_cast_impl_LINK_LIBS_DEBUG}
                ${absl_bad_any_cast_impl_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_bad_any_cast_impl_LINK_LIBS_RELEASE}
                ${absl_bad_any_cast_impl_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_bad_any_cast_impl_LINK_LIBS_RELWITHDEBINFO}
                ${absl_bad_any_cast_impl_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_bad_any_cast_impl_LINK_LIBS_MINSIZEREL}
                ${absl_bad_any_cast_impl_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::bad_any_cast_impl PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_bad_any_cast_impl_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_bad_any_cast_impl_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_bad_any_cast_impl_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_bad_any_cast_impl_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::bad_any_cast_impl PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_bad_any_cast_impl_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_bad_any_cast_impl_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_bad_any_cast_impl_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_bad_any_cast_impl_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::bad_any_cast_impl PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_bad_any_cast_impl_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_bad_any_cast_impl_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_bad_any_cast_impl_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_bad_any_cast_impl_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_bad_any_cast_impl_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_bad_any_cast_impl_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_bad_any_cast_impl_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_bad_any_cast_impl_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_bad_any_cast_impl_TARGET_PROPERTIES TRUE)

########## COMPONENT bad_any_cast TARGET PROPERTIES ######################################

set_property(TARGET absl::bad_any_cast PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_bad_any_cast_LINK_LIBS_DEBUG}
                ${absl_bad_any_cast_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_bad_any_cast_LINK_LIBS_RELEASE}
                ${absl_bad_any_cast_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_bad_any_cast_LINK_LIBS_RELWITHDEBINFO}
                ${absl_bad_any_cast_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_bad_any_cast_LINK_LIBS_MINSIZEREL}
                ${absl_bad_any_cast_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::bad_any_cast PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_bad_any_cast_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_bad_any_cast_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_bad_any_cast_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_bad_any_cast_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::bad_any_cast PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_bad_any_cast_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_bad_any_cast_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_bad_any_cast_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_bad_any_cast_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::bad_any_cast PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_bad_any_cast_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_bad_any_cast_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_bad_any_cast_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_bad_any_cast_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_bad_any_cast_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_bad_any_cast_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_bad_any_cast_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_bad_any_cast_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_bad_any_cast_TARGET_PROPERTIES TRUE)

########## COMPONENT fast_type_id TARGET PROPERTIES ######################################

set_property(TARGET absl::fast_type_id PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_fast_type_id_LINK_LIBS_DEBUG}
                ${absl_fast_type_id_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_fast_type_id_LINK_LIBS_RELEASE}
                ${absl_fast_type_id_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_fast_type_id_LINK_LIBS_RELWITHDEBINFO}
                ${absl_fast_type_id_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_fast_type_id_LINK_LIBS_MINSIZEREL}
                ${absl_fast_type_id_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::fast_type_id PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_fast_type_id_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_fast_type_id_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_fast_type_id_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_fast_type_id_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::fast_type_id PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_fast_type_id_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_fast_type_id_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_fast_type_id_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_fast_type_id_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::fast_type_id PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_fast_type_id_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_fast_type_id_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_fast_type_id_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_fast_type_id_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_fast_type_id_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_fast_type_id_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_fast_type_id_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_fast_type_id_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_fast_type_id_TARGET_PROPERTIES TRUE)

########## COMPONENT any TARGET PROPERTIES ######################################

set_property(TARGET absl::any PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_any_LINK_LIBS_DEBUG}
                ${absl_any_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_any_LINK_LIBS_RELEASE}
                ${absl_any_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_any_LINK_LIBS_RELWITHDEBINFO}
                ${absl_any_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_any_LINK_LIBS_MINSIZEREL}
                ${absl_any_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::any PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_any_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_any_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_any_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_any_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::any PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_any_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_any_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_any_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_any_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::any PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_any_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_any_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_any_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_any_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_any_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_any_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_any_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_any_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_any_TARGET_PROPERTIES TRUE)

########## COMPONENT time_zone TARGET PROPERTIES ######################################

set_property(TARGET absl::time_zone PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_time_zone_LINK_LIBS_DEBUG}
                ${absl_time_zone_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_time_zone_LINK_LIBS_RELEASE}
                ${absl_time_zone_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_time_zone_LINK_LIBS_RELWITHDEBINFO}
                ${absl_time_zone_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_time_zone_LINK_LIBS_MINSIZEREL}
                ${absl_time_zone_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::time_zone PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_time_zone_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_time_zone_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_time_zone_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_time_zone_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::time_zone PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_time_zone_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_time_zone_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_time_zone_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_time_zone_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::time_zone PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_time_zone_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_time_zone_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_time_zone_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_time_zone_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_time_zone_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_time_zone_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_time_zone_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_time_zone_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_time_zone_TARGET_PROPERTIES TRUE)

########## COMPONENT civil_time TARGET PROPERTIES ######################################

set_property(TARGET absl::civil_time PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_civil_time_LINK_LIBS_DEBUG}
                ${absl_civil_time_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_civil_time_LINK_LIBS_RELEASE}
                ${absl_civil_time_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_civil_time_LINK_LIBS_RELWITHDEBINFO}
                ${absl_civil_time_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_civil_time_LINK_LIBS_MINSIZEREL}
                ${absl_civil_time_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::civil_time PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_civil_time_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_civil_time_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_civil_time_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_civil_time_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::civil_time PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_civil_time_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_civil_time_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_civil_time_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_civil_time_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::civil_time PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_civil_time_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_civil_time_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_civil_time_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_civil_time_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_civil_time_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_civil_time_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_civil_time_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_civil_time_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_civil_time_TARGET_PROPERTIES TRUE)

########## COMPONENT dynamic_annotations TARGET PROPERTIES ######################################

set_property(TARGET absl::dynamic_annotations PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_dynamic_annotations_LINK_LIBS_DEBUG}
                ${absl_dynamic_annotations_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_dynamic_annotations_LINK_LIBS_RELEASE}
                ${absl_dynamic_annotations_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_dynamic_annotations_LINK_LIBS_RELWITHDEBINFO}
                ${absl_dynamic_annotations_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_dynamic_annotations_LINK_LIBS_MINSIZEREL}
                ${absl_dynamic_annotations_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::dynamic_annotations PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_dynamic_annotations_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_dynamic_annotations_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_dynamic_annotations_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_dynamic_annotations_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::dynamic_annotations PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_dynamic_annotations_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_dynamic_annotations_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_dynamic_annotations_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_dynamic_annotations_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::dynamic_annotations PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_dynamic_annotations_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_dynamic_annotations_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_dynamic_annotations_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_dynamic_annotations_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_dynamic_annotations_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_dynamic_annotations_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_dynamic_annotations_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_dynamic_annotations_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_dynamic_annotations_TARGET_PROPERTIES TRUE)

########## COMPONENT spinlock_wait TARGET PROPERTIES ######################################

set_property(TARGET absl::spinlock_wait PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_spinlock_wait_LINK_LIBS_DEBUG}
                ${absl_spinlock_wait_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_spinlock_wait_LINK_LIBS_RELEASE}
                ${absl_spinlock_wait_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_spinlock_wait_LINK_LIBS_RELWITHDEBINFO}
                ${absl_spinlock_wait_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_spinlock_wait_LINK_LIBS_MINSIZEREL}
                ${absl_spinlock_wait_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::spinlock_wait PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_spinlock_wait_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_spinlock_wait_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_spinlock_wait_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_spinlock_wait_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::spinlock_wait PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_spinlock_wait_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_spinlock_wait_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_spinlock_wait_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_spinlock_wait_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::spinlock_wait PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_spinlock_wait_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_spinlock_wait_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_spinlock_wait_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_spinlock_wait_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_spinlock_wait_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_spinlock_wait_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_spinlock_wait_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_spinlock_wait_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_spinlock_wait_TARGET_PROPERTIES TRUE)

########## COMPONENT base TARGET PROPERTIES ######################################

set_property(TARGET absl::base PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_base_LINK_LIBS_DEBUG}
                ${absl_base_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_base_LINK_LIBS_RELEASE}
                ${absl_base_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_base_LINK_LIBS_RELWITHDEBINFO}
                ${absl_base_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_base_LINK_LIBS_MINSIZEREL}
                ${absl_base_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::base PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_base_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_base_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_base_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_base_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::base PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_base_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_base_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_base_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_base_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::base PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_base_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_base_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_base_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_base_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_base_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_base_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_base_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_base_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_base_TARGET_PROPERTIES TRUE)

########## COMPONENT endian TARGET PROPERTIES ######################################

set_property(TARGET absl::endian PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_endian_LINK_LIBS_DEBUG}
                ${absl_endian_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_endian_LINK_LIBS_RELEASE}
                ${absl_endian_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_endian_LINK_LIBS_RELWITHDEBINFO}
                ${absl_endian_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_endian_LINK_LIBS_MINSIZEREL}
                ${absl_endian_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::endian PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_endian_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_endian_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_endian_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_endian_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::endian PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_endian_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_endian_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_endian_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_endian_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::endian PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_endian_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_endian_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_endian_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_endian_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_endian_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_endian_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_endian_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_endian_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_endian_TARGET_PROPERTIES TRUE)

########## COMPONENT strings_internal TARGET PROPERTIES ######################################

set_property(TARGET absl::strings_internal PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_strings_internal_LINK_LIBS_DEBUG}
                ${absl_strings_internal_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_strings_internal_LINK_LIBS_RELEASE}
                ${absl_strings_internal_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_strings_internal_LINK_LIBS_RELWITHDEBINFO}
                ${absl_strings_internal_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_strings_internal_LINK_LIBS_MINSIZEREL}
                ${absl_strings_internal_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::strings_internal PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_strings_internal_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_strings_internal_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_strings_internal_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_strings_internal_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::strings_internal PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_strings_internal_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_strings_internal_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_strings_internal_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_strings_internal_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::strings_internal PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_strings_internal_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_strings_internal_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_strings_internal_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_strings_internal_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_strings_internal_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_strings_internal_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_strings_internal_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_strings_internal_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_strings_internal_TARGET_PROPERTIES TRUE)

########## COMPONENT bits TARGET PROPERTIES ######################################

set_property(TARGET absl::bits PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_bits_LINK_LIBS_DEBUG}
                ${absl_bits_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_bits_LINK_LIBS_RELEASE}
                ${absl_bits_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_bits_LINK_LIBS_RELWITHDEBINFO}
                ${absl_bits_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_bits_LINK_LIBS_MINSIZEREL}
                ${absl_bits_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::bits PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_bits_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_bits_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_bits_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_bits_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::bits PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_bits_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_bits_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_bits_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_bits_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::bits PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_bits_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_bits_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_bits_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_bits_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_bits_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_bits_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_bits_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_bits_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_bits_TARGET_PROPERTIES TRUE)

########## COMPONENT int128 TARGET PROPERTIES ######################################

set_property(TARGET absl::int128 PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_int128_LINK_LIBS_DEBUG}
                ${absl_int128_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_int128_LINK_LIBS_RELEASE}
                ${absl_int128_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_int128_LINK_LIBS_RELWITHDEBINFO}
                ${absl_int128_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_int128_LINK_LIBS_MINSIZEREL}
                ${absl_int128_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::int128 PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_int128_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_int128_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_int128_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_int128_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::int128 PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_int128_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_int128_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_int128_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_int128_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::int128 PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_int128_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_int128_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_int128_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_int128_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_int128_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_int128_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_int128_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_int128_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_int128_TARGET_PROPERTIES TRUE)

########## COMPONENT strings TARGET PROPERTIES ######################################

set_property(TARGET absl::strings PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_strings_LINK_LIBS_DEBUG}
                ${absl_strings_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_strings_LINK_LIBS_RELEASE}
                ${absl_strings_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_strings_LINK_LIBS_RELWITHDEBINFO}
                ${absl_strings_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_strings_LINK_LIBS_MINSIZEREL}
                ${absl_strings_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::strings PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_strings_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_strings_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_strings_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_strings_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::strings PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_strings_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_strings_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_strings_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_strings_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::strings PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_strings_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_strings_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_strings_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_strings_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_strings_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_strings_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_strings_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_strings_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_strings_TARGET_PROPERTIES TRUE)

########## COMPONENT time TARGET PROPERTIES ######################################

set_property(TARGET absl::time PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_time_LINK_LIBS_DEBUG}
                ${absl_time_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_time_LINK_LIBS_RELEASE}
                ${absl_time_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_time_LINK_LIBS_RELWITHDEBINFO}
                ${absl_time_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_time_LINK_LIBS_MINSIZEREL}
                ${absl_time_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::time PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_time_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_time_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_time_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_time_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::time PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_time_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_time_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_time_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_time_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::time PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_time_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_time_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_time_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_time_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_time_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_time_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_time_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_time_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_time_TARGET_PROPERTIES TRUE)

########## COMPONENT kernel_timeout_internal TARGET PROPERTIES ######################################

set_property(TARGET absl::kernel_timeout_internal PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_kernel_timeout_internal_LINK_LIBS_DEBUG}
                ${absl_kernel_timeout_internal_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_kernel_timeout_internal_LINK_LIBS_RELEASE}
                ${absl_kernel_timeout_internal_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_kernel_timeout_internal_LINK_LIBS_RELWITHDEBINFO}
                ${absl_kernel_timeout_internal_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_kernel_timeout_internal_LINK_LIBS_MINSIZEREL}
                ${absl_kernel_timeout_internal_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::kernel_timeout_internal PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_kernel_timeout_internal_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_kernel_timeout_internal_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_kernel_timeout_internal_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_kernel_timeout_internal_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::kernel_timeout_internal PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_kernel_timeout_internal_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_kernel_timeout_internal_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_kernel_timeout_internal_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_kernel_timeout_internal_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::kernel_timeout_internal PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_kernel_timeout_internal_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_kernel_timeout_internal_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_kernel_timeout_internal_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_kernel_timeout_internal_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_kernel_timeout_internal_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_kernel_timeout_internal_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_kernel_timeout_internal_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_kernel_timeout_internal_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_kernel_timeout_internal_TARGET_PROPERTIES TRUE)

########## COMPONENT malloc_internal TARGET PROPERTIES ######################################

set_property(TARGET absl::malloc_internal PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_malloc_internal_LINK_LIBS_DEBUG}
                ${absl_malloc_internal_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_malloc_internal_LINK_LIBS_RELEASE}
                ${absl_malloc_internal_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_malloc_internal_LINK_LIBS_RELWITHDEBINFO}
                ${absl_malloc_internal_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_malloc_internal_LINK_LIBS_MINSIZEREL}
                ${absl_malloc_internal_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::malloc_internal PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_malloc_internal_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_malloc_internal_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_malloc_internal_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_malloc_internal_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::malloc_internal PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_malloc_internal_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_malloc_internal_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_malloc_internal_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_malloc_internal_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::malloc_internal PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_malloc_internal_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_malloc_internal_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_malloc_internal_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_malloc_internal_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_malloc_internal_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_malloc_internal_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_malloc_internal_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_malloc_internal_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_malloc_internal_TARGET_PROPERTIES TRUE)

########## COMPONENT graphcycles_internal TARGET PROPERTIES ######################################

set_property(TARGET absl::graphcycles_internal PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_graphcycles_internal_LINK_LIBS_DEBUG}
                ${absl_graphcycles_internal_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_graphcycles_internal_LINK_LIBS_RELEASE}
                ${absl_graphcycles_internal_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_graphcycles_internal_LINK_LIBS_RELWITHDEBINFO}
                ${absl_graphcycles_internal_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_graphcycles_internal_LINK_LIBS_MINSIZEREL}
                ${absl_graphcycles_internal_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::graphcycles_internal PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_graphcycles_internal_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_graphcycles_internal_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_graphcycles_internal_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_graphcycles_internal_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::graphcycles_internal PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_graphcycles_internal_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_graphcycles_internal_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_graphcycles_internal_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_graphcycles_internal_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::graphcycles_internal PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_graphcycles_internal_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_graphcycles_internal_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_graphcycles_internal_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_graphcycles_internal_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_graphcycles_internal_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_graphcycles_internal_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_graphcycles_internal_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_graphcycles_internal_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_graphcycles_internal_TARGET_PROPERTIES TRUE)

########## COMPONENT demangle_internal TARGET PROPERTIES ######################################

set_property(TARGET absl::demangle_internal PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_demangle_internal_LINK_LIBS_DEBUG}
                ${absl_demangle_internal_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_demangle_internal_LINK_LIBS_RELEASE}
                ${absl_demangle_internal_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_demangle_internal_LINK_LIBS_RELWITHDEBINFO}
                ${absl_demangle_internal_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_demangle_internal_LINK_LIBS_MINSIZEREL}
                ${absl_demangle_internal_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::demangle_internal PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_demangle_internal_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_demangle_internal_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_demangle_internal_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_demangle_internal_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::demangle_internal PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_demangle_internal_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_demangle_internal_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_demangle_internal_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_demangle_internal_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::demangle_internal PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_demangle_internal_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_demangle_internal_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_demangle_internal_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_demangle_internal_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_demangle_internal_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_demangle_internal_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_demangle_internal_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_demangle_internal_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_demangle_internal_TARGET_PROPERTIES TRUE)

########## COMPONENT debugging_internal TARGET PROPERTIES ######################################

set_property(TARGET absl::debugging_internal PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_debugging_internal_LINK_LIBS_DEBUG}
                ${absl_debugging_internal_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_debugging_internal_LINK_LIBS_RELEASE}
                ${absl_debugging_internal_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_debugging_internal_LINK_LIBS_RELWITHDEBINFO}
                ${absl_debugging_internal_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_debugging_internal_LINK_LIBS_MINSIZEREL}
                ${absl_debugging_internal_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::debugging_internal PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_debugging_internal_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_debugging_internal_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_debugging_internal_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_debugging_internal_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::debugging_internal PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_debugging_internal_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_debugging_internal_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_debugging_internal_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_debugging_internal_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::debugging_internal PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_debugging_internal_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_debugging_internal_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_debugging_internal_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_debugging_internal_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_debugging_internal_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_debugging_internal_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_debugging_internal_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_debugging_internal_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_debugging_internal_TARGET_PROPERTIES TRUE)

########## COMPONENT symbolize TARGET PROPERTIES ######################################

set_property(TARGET absl::symbolize PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_symbolize_LINK_LIBS_DEBUG}
                ${absl_symbolize_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_symbolize_LINK_LIBS_RELEASE}
                ${absl_symbolize_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_symbolize_LINK_LIBS_RELWITHDEBINFO}
                ${absl_symbolize_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_symbolize_LINK_LIBS_MINSIZEREL}
                ${absl_symbolize_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::symbolize PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_symbolize_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_symbolize_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_symbolize_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_symbolize_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::symbolize PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_symbolize_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_symbolize_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_symbolize_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_symbolize_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::symbolize PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_symbolize_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_symbolize_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_symbolize_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_symbolize_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_symbolize_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_symbolize_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_symbolize_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_symbolize_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_symbolize_TARGET_PROPERTIES TRUE)

########## COMPONENT stacktrace TARGET PROPERTIES ######################################

set_property(TARGET absl::stacktrace PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_stacktrace_LINK_LIBS_DEBUG}
                ${absl_stacktrace_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_stacktrace_LINK_LIBS_RELEASE}
                ${absl_stacktrace_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_stacktrace_LINK_LIBS_RELWITHDEBINFO}
                ${absl_stacktrace_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_stacktrace_LINK_LIBS_MINSIZEREL}
                ${absl_stacktrace_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::stacktrace PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_stacktrace_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_stacktrace_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_stacktrace_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_stacktrace_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::stacktrace PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_stacktrace_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_stacktrace_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_stacktrace_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_stacktrace_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::stacktrace PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_stacktrace_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_stacktrace_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_stacktrace_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_stacktrace_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_stacktrace_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_stacktrace_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_stacktrace_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_stacktrace_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_stacktrace_TARGET_PROPERTIES TRUE)

########## COMPONENT synchronization TARGET PROPERTIES ######################################

set_property(TARGET absl::synchronization PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_synchronization_LINK_LIBS_DEBUG}
                ${absl_synchronization_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_synchronization_LINK_LIBS_RELEASE}
                ${absl_synchronization_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_synchronization_LINK_LIBS_RELWITHDEBINFO}
                ${absl_synchronization_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_synchronization_LINK_LIBS_MINSIZEREL}
                ${absl_synchronization_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::synchronization PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_synchronization_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_synchronization_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_synchronization_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_synchronization_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::synchronization PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_synchronization_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_synchronization_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_synchronization_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_synchronization_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::synchronization PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_synchronization_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_synchronization_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_synchronization_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_synchronization_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_synchronization_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_synchronization_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_synchronization_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_synchronization_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_synchronization_TARGET_PROPERTIES TRUE)

########## COMPONENT cordz_handle TARGET PROPERTIES ######################################

set_property(TARGET absl::cordz_handle PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_cordz_handle_LINK_LIBS_DEBUG}
                ${absl_cordz_handle_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_cordz_handle_LINK_LIBS_RELEASE}
                ${absl_cordz_handle_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_cordz_handle_LINK_LIBS_RELWITHDEBINFO}
                ${absl_cordz_handle_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_cordz_handle_LINK_LIBS_MINSIZEREL}
                ${absl_cordz_handle_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::cordz_handle PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_cordz_handle_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_cordz_handle_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_cordz_handle_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_cordz_handle_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::cordz_handle PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_cordz_handle_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_cordz_handle_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_cordz_handle_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_cordz_handle_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::cordz_handle PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_cordz_handle_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_cordz_handle_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_cordz_handle_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_cordz_handle_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_cordz_handle_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_cordz_handle_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_cordz_handle_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_cordz_handle_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_cordz_handle_TARGET_PROPERTIES TRUE)

########## COMPONENT cordz_update_tracker TARGET PROPERTIES ######################################

set_property(TARGET absl::cordz_update_tracker PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_cordz_update_tracker_LINK_LIBS_DEBUG}
                ${absl_cordz_update_tracker_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_cordz_update_tracker_LINK_LIBS_RELEASE}
                ${absl_cordz_update_tracker_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_cordz_update_tracker_LINK_LIBS_RELWITHDEBINFO}
                ${absl_cordz_update_tracker_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_cordz_update_tracker_LINK_LIBS_MINSIZEREL}
                ${absl_cordz_update_tracker_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::cordz_update_tracker PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_cordz_update_tracker_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_cordz_update_tracker_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_cordz_update_tracker_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_cordz_update_tracker_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::cordz_update_tracker PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_cordz_update_tracker_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_cordz_update_tracker_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_cordz_update_tracker_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_cordz_update_tracker_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::cordz_update_tracker PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_cordz_update_tracker_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_cordz_update_tracker_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_cordz_update_tracker_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_cordz_update_tracker_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_cordz_update_tracker_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_cordz_update_tracker_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_cordz_update_tracker_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_cordz_update_tracker_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_cordz_update_tracker_TARGET_PROPERTIES TRUE)

########## COMPONENT cordz_statistics TARGET PROPERTIES ######################################

set_property(TARGET absl::cordz_statistics PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_cordz_statistics_LINK_LIBS_DEBUG}
                ${absl_cordz_statistics_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_cordz_statistics_LINK_LIBS_RELEASE}
                ${absl_cordz_statistics_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_cordz_statistics_LINK_LIBS_RELWITHDEBINFO}
                ${absl_cordz_statistics_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_cordz_statistics_LINK_LIBS_MINSIZEREL}
                ${absl_cordz_statistics_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::cordz_statistics PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_cordz_statistics_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_cordz_statistics_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_cordz_statistics_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_cordz_statistics_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::cordz_statistics PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_cordz_statistics_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_cordz_statistics_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_cordz_statistics_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_cordz_statistics_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::cordz_statistics PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_cordz_statistics_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_cordz_statistics_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_cordz_statistics_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_cordz_statistics_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_cordz_statistics_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_cordz_statistics_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_cordz_statistics_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_cordz_statistics_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_cordz_statistics_TARGET_PROPERTIES TRUE)

########## COMPONENT exponential_biased TARGET PROPERTIES ######################################

set_property(TARGET absl::exponential_biased PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_exponential_biased_LINK_LIBS_DEBUG}
                ${absl_exponential_biased_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_exponential_biased_LINK_LIBS_RELEASE}
                ${absl_exponential_biased_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_exponential_biased_LINK_LIBS_RELWITHDEBINFO}
                ${absl_exponential_biased_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_exponential_biased_LINK_LIBS_MINSIZEREL}
                ${absl_exponential_biased_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::exponential_biased PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_exponential_biased_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_exponential_biased_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_exponential_biased_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_exponential_biased_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::exponential_biased PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_exponential_biased_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_exponential_biased_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_exponential_biased_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_exponential_biased_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::exponential_biased PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_exponential_biased_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_exponential_biased_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_exponential_biased_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_exponential_biased_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_exponential_biased_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_exponential_biased_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_exponential_biased_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_exponential_biased_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_exponential_biased_TARGET_PROPERTIES TRUE)

########## COMPONENT cordz_functions TARGET PROPERTIES ######################################

set_property(TARGET absl::cordz_functions PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_cordz_functions_LINK_LIBS_DEBUG}
                ${absl_cordz_functions_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_cordz_functions_LINK_LIBS_RELEASE}
                ${absl_cordz_functions_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_cordz_functions_LINK_LIBS_RELWITHDEBINFO}
                ${absl_cordz_functions_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_cordz_functions_LINK_LIBS_MINSIZEREL}
                ${absl_cordz_functions_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::cordz_functions PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_cordz_functions_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_cordz_functions_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_cordz_functions_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_cordz_functions_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::cordz_functions PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_cordz_functions_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_cordz_functions_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_cordz_functions_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_cordz_functions_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::cordz_functions PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_cordz_functions_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_cordz_functions_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_cordz_functions_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_cordz_functions_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_cordz_functions_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_cordz_functions_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_cordz_functions_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_cordz_functions_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_cordz_functions_TARGET_PROPERTIES TRUE)

########## COMPONENT non_temporal_arm_intrinsics TARGET PROPERTIES ######################################

set_property(TARGET absl::non_temporal_arm_intrinsics PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_non_temporal_arm_intrinsics_LINK_LIBS_DEBUG}
                ${absl_non_temporal_arm_intrinsics_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_non_temporal_arm_intrinsics_LINK_LIBS_RELEASE}
                ${absl_non_temporal_arm_intrinsics_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_non_temporal_arm_intrinsics_LINK_LIBS_RELWITHDEBINFO}
                ${absl_non_temporal_arm_intrinsics_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_non_temporal_arm_intrinsics_LINK_LIBS_MINSIZEREL}
                ${absl_non_temporal_arm_intrinsics_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::non_temporal_arm_intrinsics PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_non_temporal_arm_intrinsics_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_non_temporal_arm_intrinsics_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_non_temporal_arm_intrinsics_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_non_temporal_arm_intrinsics_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::non_temporal_arm_intrinsics PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_non_temporal_arm_intrinsics_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_non_temporal_arm_intrinsics_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_non_temporal_arm_intrinsics_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_non_temporal_arm_intrinsics_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::non_temporal_arm_intrinsics PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_non_temporal_arm_intrinsics_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_non_temporal_arm_intrinsics_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_non_temporal_arm_intrinsics_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_non_temporal_arm_intrinsics_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_non_temporal_arm_intrinsics_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_non_temporal_arm_intrinsics_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_non_temporal_arm_intrinsics_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_non_temporal_arm_intrinsics_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_non_temporal_arm_intrinsics_TARGET_PROPERTIES TRUE)

########## COMPONENT non_temporal_memcpy TARGET PROPERTIES ######################################

set_property(TARGET absl::non_temporal_memcpy PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_non_temporal_memcpy_LINK_LIBS_DEBUG}
                ${absl_non_temporal_memcpy_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_non_temporal_memcpy_LINK_LIBS_RELEASE}
                ${absl_non_temporal_memcpy_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_non_temporal_memcpy_LINK_LIBS_RELWITHDEBINFO}
                ${absl_non_temporal_memcpy_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_non_temporal_memcpy_LINK_LIBS_MINSIZEREL}
                ${absl_non_temporal_memcpy_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::non_temporal_memcpy PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_non_temporal_memcpy_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_non_temporal_memcpy_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_non_temporal_memcpy_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_non_temporal_memcpy_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::non_temporal_memcpy PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_non_temporal_memcpy_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_non_temporal_memcpy_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_non_temporal_memcpy_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_non_temporal_memcpy_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::non_temporal_memcpy PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_non_temporal_memcpy_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_non_temporal_memcpy_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_non_temporal_memcpy_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_non_temporal_memcpy_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_non_temporal_memcpy_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_non_temporal_memcpy_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_non_temporal_memcpy_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_non_temporal_memcpy_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_non_temporal_memcpy_TARGET_PROPERTIES TRUE)

########## COMPONENT crc_cpu_detect TARGET PROPERTIES ######################################

set_property(TARGET absl::crc_cpu_detect PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_crc_cpu_detect_LINK_LIBS_DEBUG}
                ${absl_crc_cpu_detect_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_crc_cpu_detect_LINK_LIBS_RELEASE}
                ${absl_crc_cpu_detect_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_crc_cpu_detect_LINK_LIBS_RELWITHDEBINFO}
                ${absl_crc_cpu_detect_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_crc_cpu_detect_LINK_LIBS_MINSIZEREL}
                ${absl_crc_cpu_detect_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::crc_cpu_detect PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_crc_cpu_detect_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_crc_cpu_detect_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_crc_cpu_detect_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_crc_cpu_detect_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::crc_cpu_detect PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_crc_cpu_detect_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_crc_cpu_detect_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_crc_cpu_detect_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_crc_cpu_detect_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::crc_cpu_detect PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_crc_cpu_detect_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_crc_cpu_detect_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_crc_cpu_detect_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_crc_cpu_detect_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_crc_cpu_detect_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_crc_cpu_detect_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_crc_cpu_detect_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_crc_cpu_detect_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_crc_cpu_detect_TARGET_PROPERTIES TRUE)

########## COMPONENT prefetch TARGET PROPERTIES ######################################

set_property(TARGET absl::prefetch PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_prefetch_LINK_LIBS_DEBUG}
                ${absl_prefetch_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_prefetch_LINK_LIBS_RELEASE}
                ${absl_prefetch_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_prefetch_LINK_LIBS_RELWITHDEBINFO}
                ${absl_prefetch_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_prefetch_LINK_LIBS_MINSIZEREL}
                ${absl_prefetch_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::prefetch PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_prefetch_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_prefetch_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_prefetch_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_prefetch_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::prefetch PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_prefetch_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_prefetch_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_prefetch_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_prefetch_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::prefetch PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_prefetch_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_prefetch_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_prefetch_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_prefetch_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_prefetch_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_prefetch_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_prefetch_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_prefetch_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_prefetch_TARGET_PROPERTIES TRUE)

########## COMPONENT crc_internal TARGET PROPERTIES ######################################

set_property(TARGET absl::crc_internal PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_crc_internal_LINK_LIBS_DEBUG}
                ${absl_crc_internal_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_crc_internal_LINK_LIBS_RELEASE}
                ${absl_crc_internal_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_crc_internal_LINK_LIBS_RELWITHDEBINFO}
                ${absl_crc_internal_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_crc_internal_LINK_LIBS_MINSIZEREL}
                ${absl_crc_internal_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::crc_internal PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_crc_internal_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_crc_internal_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_crc_internal_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_crc_internal_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::crc_internal PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_crc_internal_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_crc_internal_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_crc_internal_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_crc_internal_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::crc_internal PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_crc_internal_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_crc_internal_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_crc_internal_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_crc_internal_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_crc_internal_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_crc_internal_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_crc_internal_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_crc_internal_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_crc_internal_TARGET_PROPERTIES TRUE)

########## COMPONENT crc32c TARGET PROPERTIES ######################################

set_property(TARGET absl::crc32c PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_crc32c_LINK_LIBS_DEBUG}
                ${absl_crc32c_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_crc32c_LINK_LIBS_RELEASE}
                ${absl_crc32c_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_crc32c_LINK_LIBS_RELWITHDEBINFO}
                ${absl_crc32c_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_crc32c_LINK_LIBS_MINSIZEREL}
                ${absl_crc32c_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::crc32c PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_crc32c_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_crc32c_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_crc32c_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_crc32c_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::crc32c PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_crc32c_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_crc32c_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_crc32c_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_crc32c_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::crc32c PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_crc32c_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_crc32c_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_crc32c_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_crc32c_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_crc32c_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_crc32c_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_crc32c_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_crc32c_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_crc32c_TARGET_PROPERTIES TRUE)

########## COMPONENT crc_cord_state TARGET PROPERTIES ######################################

set_property(TARGET absl::crc_cord_state PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_crc_cord_state_LINK_LIBS_DEBUG}
                ${absl_crc_cord_state_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_crc_cord_state_LINK_LIBS_RELEASE}
                ${absl_crc_cord_state_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_crc_cord_state_LINK_LIBS_RELWITHDEBINFO}
                ${absl_crc_cord_state_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_crc_cord_state_LINK_LIBS_MINSIZEREL}
                ${absl_crc_cord_state_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::crc_cord_state PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_crc_cord_state_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_crc_cord_state_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_crc_cord_state_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_crc_cord_state_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::crc_cord_state PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_crc_cord_state_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_crc_cord_state_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_crc_cord_state_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_crc_cord_state_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::crc_cord_state PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_crc_cord_state_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_crc_cord_state_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_crc_cord_state_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_crc_cord_state_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_crc_cord_state_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_crc_cord_state_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_crc_cord_state_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_crc_cord_state_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_crc_cord_state_TARGET_PROPERTIES TRUE)

########## COMPONENT layout TARGET PROPERTIES ######################################

set_property(TARGET absl::layout PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_layout_LINK_LIBS_DEBUG}
                ${absl_layout_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_layout_LINK_LIBS_RELEASE}
                ${absl_layout_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_layout_LINK_LIBS_RELWITHDEBINFO}
                ${absl_layout_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_layout_LINK_LIBS_MINSIZEREL}
                ${absl_layout_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::layout PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_layout_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_layout_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_layout_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_layout_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::layout PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_layout_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_layout_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_layout_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_layout_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::layout PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_layout_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_layout_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_layout_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_layout_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_layout_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_layout_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_layout_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_layout_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_layout_TARGET_PROPERTIES TRUE)

########## COMPONENT container_memory TARGET PROPERTIES ######################################

set_property(TARGET absl::container_memory PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_container_memory_LINK_LIBS_DEBUG}
                ${absl_container_memory_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_container_memory_LINK_LIBS_RELEASE}
                ${absl_container_memory_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_container_memory_LINK_LIBS_RELWITHDEBINFO}
                ${absl_container_memory_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_container_memory_LINK_LIBS_MINSIZEREL}
                ${absl_container_memory_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::container_memory PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_container_memory_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_container_memory_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_container_memory_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_container_memory_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::container_memory PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_container_memory_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_container_memory_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_container_memory_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_container_memory_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::container_memory PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_container_memory_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_container_memory_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_container_memory_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_container_memory_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_container_memory_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_container_memory_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_container_memory_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_container_memory_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_container_memory_TARGET_PROPERTIES TRUE)

########## COMPONENT compressed_tuple TARGET PROPERTIES ######################################

set_property(TARGET absl::compressed_tuple PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_compressed_tuple_LINK_LIBS_DEBUG}
                ${absl_compressed_tuple_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_compressed_tuple_LINK_LIBS_RELEASE}
                ${absl_compressed_tuple_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_compressed_tuple_LINK_LIBS_RELWITHDEBINFO}
                ${absl_compressed_tuple_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_compressed_tuple_LINK_LIBS_MINSIZEREL}
                ${absl_compressed_tuple_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::compressed_tuple PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_compressed_tuple_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_compressed_tuple_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_compressed_tuple_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_compressed_tuple_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::compressed_tuple PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_compressed_tuple_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_compressed_tuple_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_compressed_tuple_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_compressed_tuple_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::compressed_tuple PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_compressed_tuple_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_compressed_tuple_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_compressed_tuple_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_compressed_tuple_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_compressed_tuple_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_compressed_tuple_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_compressed_tuple_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_compressed_tuple_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_compressed_tuple_TARGET_PROPERTIES TRUE)

########## COMPONENT inlined_vector_internal TARGET PROPERTIES ######################################

set_property(TARGET absl::inlined_vector_internal PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_inlined_vector_internal_LINK_LIBS_DEBUG}
                ${absl_inlined_vector_internal_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_inlined_vector_internal_LINK_LIBS_RELEASE}
                ${absl_inlined_vector_internal_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_inlined_vector_internal_LINK_LIBS_RELWITHDEBINFO}
                ${absl_inlined_vector_internal_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_inlined_vector_internal_LINK_LIBS_MINSIZEREL}
                ${absl_inlined_vector_internal_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::inlined_vector_internal PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_inlined_vector_internal_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_inlined_vector_internal_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_inlined_vector_internal_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_inlined_vector_internal_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::inlined_vector_internal PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_inlined_vector_internal_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_inlined_vector_internal_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_inlined_vector_internal_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_inlined_vector_internal_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::inlined_vector_internal PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_inlined_vector_internal_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_inlined_vector_internal_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_inlined_vector_internal_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_inlined_vector_internal_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_inlined_vector_internal_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_inlined_vector_internal_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_inlined_vector_internal_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_inlined_vector_internal_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_inlined_vector_internal_TARGET_PROPERTIES TRUE)

########## COMPONENT inlined_vector TARGET PROPERTIES ######################################

set_property(TARGET absl::inlined_vector PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_inlined_vector_LINK_LIBS_DEBUG}
                ${absl_inlined_vector_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_inlined_vector_LINK_LIBS_RELEASE}
                ${absl_inlined_vector_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_inlined_vector_LINK_LIBS_RELWITHDEBINFO}
                ${absl_inlined_vector_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_inlined_vector_LINK_LIBS_MINSIZEREL}
                ${absl_inlined_vector_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::inlined_vector PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_inlined_vector_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_inlined_vector_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_inlined_vector_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_inlined_vector_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::inlined_vector PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_inlined_vector_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_inlined_vector_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_inlined_vector_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_inlined_vector_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::inlined_vector PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_inlined_vector_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_inlined_vector_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_inlined_vector_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_inlined_vector_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_inlined_vector_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_inlined_vector_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_inlined_vector_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_inlined_vector_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_inlined_vector_TARGET_PROPERTIES TRUE)

########## COMPONENT cord_internal TARGET PROPERTIES ######################################

set_property(TARGET absl::cord_internal PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_cord_internal_LINK_LIBS_DEBUG}
                ${absl_cord_internal_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_cord_internal_LINK_LIBS_RELEASE}
                ${absl_cord_internal_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_cord_internal_LINK_LIBS_RELWITHDEBINFO}
                ${absl_cord_internal_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_cord_internal_LINK_LIBS_MINSIZEREL}
                ${absl_cord_internal_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::cord_internal PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_cord_internal_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_cord_internal_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_cord_internal_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_cord_internal_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::cord_internal PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_cord_internal_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_cord_internal_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_cord_internal_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_cord_internal_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::cord_internal PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_cord_internal_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_cord_internal_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_cord_internal_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_cord_internal_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_cord_internal_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_cord_internal_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_cord_internal_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_cord_internal_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_cord_internal_TARGET_PROPERTIES TRUE)

########## COMPONENT cordz_info TARGET PROPERTIES ######################################

set_property(TARGET absl::cordz_info PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_cordz_info_LINK_LIBS_DEBUG}
                ${absl_cordz_info_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_cordz_info_LINK_LIBS_RELEASE}
                ${absl_cordz_info_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_cordz_info_LINK_LIBS_RELWITHDEBINFO}
                ${absl_cordz_info_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_cordz_info_LINK_LIBS_MINSIZEREL}
                ${absl_cordz_info_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::cordz_info PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_cordz_info_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_cordz_info_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_cordz_info_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_cordz_info_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::cordz_info PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_cordz_info_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_cordz_info_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_cordz_info_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_cordz_info_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::cordz_info PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_cordz_info_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_cordz_info_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_cordz_info_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_cordz_info_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_cordz_info_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_cordz_info_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_cordz_info_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_cordz_info_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_cordz_info_TARGET_PROPERTIES TRUE)

########## COMPONENT cordz_update_scope TARGET PROPERTIES ######################################

set_property(TARGET absl::cordz_update_scope PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_cordz_update_scope_LINK_LIBS_DEBUG}
                ${absl_cordz_update_scope_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_cordz_update_scope_LINK_LIBS_RELEASE}
                ${absl_cordz_update_scope_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_cordz_update_scope_LINK_LIBS_RELWITHDEBINFO}
                ${absl_cordz_update_scope_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_cordz_update_scope_LINK_LIBS_MINSIZEREL}
                ${absl_cordz_update_scope_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::cordz_update_scope PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_cordz_update_scope_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_cordz_update_scope_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_cordz_update_scope_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_cordz_update_scope_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::cordz_update_scope PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_cordz_update_scope_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_cordz_update_scope_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_cordz_update_scope_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_cordz_update_scope_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::cordz_update_scope PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_cordz_update_scope_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_cordz_update_scope_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_cordz_update_scope_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_cordz_update_scope_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_cordz_update_scope_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_cordz_update_scope_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_cordz_update_scope_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_cordz_update_scope_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_cordz_update_scope_TARGET_PROPERTIES TRUE)

########## COMPONENT function_ref TARGET PROPERTIES ######################################

set_property(TARGET absl::function_ref PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_function_ref_LINK_LIBS_DEBUG}
                ${absl_function_ref_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_function_ref_LINK_LIBS_RELEASE}
                ${absl_function_ref_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_function_ref_LINK_LIBS_RELWITHDEBINFO}
                ${absl_function_ref_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_function_ref_LINK_LIBS_MINSIZEREL}
                ${absl_function_ref_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::function_ref PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_function_ref_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_function_ref_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_function_ref_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_function_ref_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::function_ref PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_function_ref_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_function_ref_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_function_ref_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_function_ref_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::function_ref PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_function_ref_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_function_ref_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_function_ref_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_function_ref_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_function_ref_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_function_ref_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_function_ref_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_function_ref_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_function_ref_TARGET_PROPERTIES TRUE)

########## COMPONENT fixed_array TARGET PROPERTIES ######################################

set_property(TARGET absl::fixed_array PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_fixed_array_LINK_LIBS_DEBUG}
                ${absl_fixed_array_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_fixed_array_LINK_LIBS_RELEASE}
                ${absl_fixed_array_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_fixed_array_LINK_LIBS_RELWITHDEBINFO}
                ${absl_fixed_array_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_fixed_array_LINK_LIBS_MINSIZEREL}
                ${absl_fixed_array_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::fixed_array PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_fixed_array_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_fixed_array_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_fixed_array_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_fixed_array_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::fixed_array PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_fixed_array_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_fixed_array_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_fixed_array_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_fixed_array_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::fixed_array PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_fixed_array_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_fixed_array_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_fixed_array_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_fixed_array_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_fixed_array_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_fixed_array_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_fixed_array_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_fixed_array_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_fixed_array_TARGET_PROPERTIES TRUE)

########## COMPONENT cord TARGET PROPERTIES ######################################

set_property(TARGET absl::cord PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_cord_LINK_LIBS_DEBUG}
                ${absl_cord_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_cord_LINK_LIBS_RELEASE}
                ${absl_cord_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_cord_LINK_LIBS_RELWITHDEBINFO}
                ${absl_cord_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_cord_LINK_LIBS_MINSIZEREL}
                ${absl_cord_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::cord PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_cord_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_cord_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_cord_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_cord_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::cord PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_cord_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_cord_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_cord_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_cord_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::cord PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_cord_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_cord_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_cord_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_cord_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_cord_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_cord_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_cord_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_cord_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_cord_TARGET_PROPERTIES TRUE)

########## COMPONENT cordz_sample_token TARGET PROPERTIES ######################################

set_property(TARGET absl::cordz_sample_token PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_cordz_sample_token_LINK_LIBS_DEBUG}
                ${absl_cordz_sample_token_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_cordz_sample_token_LINK_LIBS_RELEASE}
                ${absl_cordz_sample_token_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_cordz_sample_token_LINK_LIBS_RELWITHDEBINFO}
                ${absl_cordz_sample_token_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_cordz_sample_token_LINK_LIBS_MINSIZEREL}
                ${absl_cordz_sample_token_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::cordz_sample_token PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_cordz_sample_token_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_cordz_sample_token_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_cordz_sample_token_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_cordz_sample_token_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::cordz_sample_token PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_cordz_sample_token_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_cordz_sample_token_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_cordz_sample_token_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_cordz_sample_token_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::cordz_sample_token PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_cordz_sample_token_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_cordz_sample_token_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_cordz_sample_token_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_cordz_sample_token_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_cordz_sample_token_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_cordz_sample_token_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_cordz_sample_token_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_cordz_sample_token_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_cordz_sample_token_TARGET_PROPERTIES TRUE)

########## COMPONENT numeric_representation TARGET PROPERTIES ######################################

set_property(TARGET absl::numeric_representation PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_numeric_representation_LINK_LIBS_DEBUG}
                ${absl_numeric_representation_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_numeric_representation_LINK_LIBS_RELEASE}
                ${absl_numeric_representation_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_numeric_representation_LINK_LIBS_RELWITHDEBINFO}
                ${absl_numeric_representation_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_numeric_representation_LINK_LIBS_MINSIZEREL}
                ${absl_numeric_representation_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::numeric_representation PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_numeric_representation_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_numeric_representation_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_numeric_representation_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_numeric_representation_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::numeric_representation PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_numeric_representation_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_numeric_representation_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_numeric_representation_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_numeric_representation_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::numeric_representation PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_numeric_representation_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_numeric_representation_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_numeric_representation_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_numeric_representation_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_numeric_representation_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_numeric_representation_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_numeric_representation_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_numeric_representation_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_numeric_representation_TARGET_PROPERTIES TRUE)

########## COMPONENT str_format_internal TARGET PROPERTIES ######################################

set_property(TARGET absl::str_format_internal PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_str_format_internal_LINK_LIBS_DEBUG}
                ${absl_str_format_internal_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_str_format_internal_LINK_LIBS_RELEASE}
                ${absl_str_format_internal_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_str_format_internal_LINK_LIBS_RELWITHDEBINFO}
                ${absl_str_format_internal_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_str_format_internal_LINK_LIBS_MINSIZEREL}
                ${absl_str_format_internal_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::str_format_internal PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_str_format_internal_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_str_format_internal_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_str_format_internal_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_str_format_internal_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::str_format_internal PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_str_format_internal_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_str_format_internal_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_str_format_internal_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_str_format_internal_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::str_format_internal PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_str_format_internal_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_str_format_internal_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_str_format_internal_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_str_format_internal_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_str_format_internal_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_str_format_internal_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_str_format_internal_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_str_format_internal_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_str_format_internal_TARGET_PROPERTIES TRUE)

########## COMPONENT str_format TARGET PROPERTIES ######################################

set_property(TARGET absl::str_format PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_str_format_LINK_LIBS_DEBUG}
                ${absl_str_format_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_str_format_LINK_LIBS_RELEASE}
                ${absl_str_format_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_str_format_LINK_LIBS_RELWITHDEBINFO}
                ${absl_str_format_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_str_format_LINK_LIBS_MINSIZEREL}
                ${absl_str_format_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::str_format PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_str_format_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_str_format_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_str_format_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_str_format_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::str_format PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_str_format_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_str_format_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_str_format_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_str_format_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::str_format PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_str_format_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_str_format_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_str_format_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_str_format_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_str_format_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_str_format_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_str_format_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_str_format_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_str_format_TARGET_PROPERTIES TRUE)

########## COMPONENT strerror TARGET PROPERTIES ######################################

set_property(TARGET absl::strerror PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_strerror_LINK_LIBS_DEBUG}
                ${absl_strerror_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_strerror_LINK_LIBS_RELEASE}
                ${absl_strerror_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_strerror_LINK_LIBS_RELWITHDEBINFO}
                ${absl_strerror_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_strerror_LINK_LIBS_MINSIZEREL}
                ${absl_strerror_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::strerror PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_strerror_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_strerror_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_strerror_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_strerror_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::strerror PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_strerror_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_strerror_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_strerror_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_strerror_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::strerror PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_strerror_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_strerror_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_strerror_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_strerror_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_strerror_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_strerror_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_strerror_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_strerror_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_strerror_TARGET_PROPERTIES TRUE)

########## COMPONENT status TARGET PROPERTIES ######################################

set_property(TARGET absl::status PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_status_LINK_LIBS_DEBUG}
                ${absl_status_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_status_LINK_LIBS_RELEASE}
                ${absl_status_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_status_LINK_LIBS_RELWITHDEBINFO}
                ${absl_status_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_status_LINK_LIBS_MINSIZEREL}
                ${absl_status_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::status PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_status_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_status_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_status_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_status_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::status PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_status_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_status_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_status_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_status_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::status PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_status_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_status_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_status_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_status_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_status_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_status_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_status_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_status_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_status_TARGET_PROPERTIES TRUE)

########## COMPONENT statusor TARGET PROPERTIES ######################################

set_property(TARGET absl::statusor PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_statusor_LINK_LIBS_DEBUG}
                ${absl_statusor_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_statusor_LINK_LIBS_RELEASE}
                ${absl_statusor_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_statusor_LINK_LIBS_RELWITHDEBINFO}
                ${absl_statusor_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_statusor_LINK_LIBS_MINSIZEREL}
                ${absl_statusor_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::statusor PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_statusor_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_statusor_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_statusor_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_statusor_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::statusor PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_statusor_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_statusor_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_statusor_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_statusor_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::statusor PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_statusor_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_statusor_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_statusor_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_statusor_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_statusor_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_statusor_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_statusor_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_statusor_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_statusor_TARGET_PROPERTIES TRUE)

########## COMPONENT random_internal_traits TARGET PROPERTIES ######################################

set_property(TARGET absl::random_internal_traits PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_random_internal_traits_LINK_LIBS_DEBUG}
                ${absl_random_internal_traits_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_random_internal_traits_LINK_LIBS_RELEASE}
                ${absl_random_internal_traits_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_random_internal_traits_LINK_LIBS_RELWITHDEBINFO}
                ${absl_random_internal_traits_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_random_internal_traits_LINK_LIBS_MINSIZEREL}
                ${absl_random_internal_traits_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::random_internal_traits PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_random_internal_traits_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_random_internal_traits_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_random_internal_traits_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_random_internal_traits_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::random_internal_traits PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_random_internal_traits_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_random_internal_traits_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_random_internal_traits_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_random_internal_traits_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::random_internal_traits PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_random_internal_traits_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_random_internal_traits_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_random_internal_traits_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_random_internal_traits_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_random_internal_traits_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_random_internal_traits_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_random_internal_traits_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_random_internal_traits_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_random_internal_traits_TARGET_PROPERTIES TRUE)

########## COMPONENT random_internal_uniform_helper TARGET PROPERTIES ######################################

set_property(TARGET absl::random_internal_uniform_helper PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_random_internal_uniform_helper_LINK_LIBS_DEBUG}
                ${absl_random_internal_uniform_helper_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_random_internal_uniform_helper_LINK_LIBS_RELEASE}
                ${absl_random_internal_uniform_helper_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_random_internal_uniform_helper_LINK_LIBS_RELWITHDEBINFO}
                ${absl_random_internal_uniform_helper_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_random_internal_uniform_helper_LINK_LIBS_MINSIZEREL}
                ${absl_random_internal_uniform_helper_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::random_internal_uniform_helper PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_random_internal_uniform_helper_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_random_internal_uniform_helper_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_random_internal_uniform_helper_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_random_internal_uniform_helper_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::random_internal_uniform_helper PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_random_internal_uniform_helper_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_random_internal_uniform_helper_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_random_internal_uniform_helper_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_random_internal_uniform_helper_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::random_internal_uniform_helper PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_random_internal_uniform_helper_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_random_internal_uniform_helper_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_random_internal_uniform_helper_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_random_internal_uniform_helper_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_random_internal_uniform_helper_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_random_internal_uniform_helper_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_random_internal_uniform_helper_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_random_internal_uniform_helper_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_random_internal_uniform_helper_TARGET_PROPERTIES TRUE)

########## COMPONENT random_internal_distribution_test_util TARGET PROPERTIES ######################################

set_property(TARGET absl::random_internal_distribution_test_util PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_random_internal_distribution_test_util_LINK_LIBS_DEBUG}
                ${absl_random_internal_distribution_test_util_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_random_internal_distribution_test_util_LINK_LIBS_RELEASE}
                ${absl_random_internal_distribution_test_util_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_random_internal_distribution_test_util_LINK_LIBS_RELWITHDEBINFO}
                ${absl_random_internal_distribution_test_util_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_random_internal_distribution_test_util_LINK_LIBS_MINSIZEREL}
                ${absl_random_internal_distribution_test_util_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::random_internal_distribution_test_util PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_random_internal_distribution_test_util_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_random_internal_distribution_test_util_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_random_internal_distribution_test_util_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_random_internal_distribution_test_util_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::random_internal_distribution_test_util PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_random_internal_distribution_test_util_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_random_internal_distribution_test_util_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_random_internal_distribution_test_util_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_random_internal_distribution_test_util_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::random_internal_distribution_test_util PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_random_internal_distribution_test_util_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_random_internal_distribution_test_util_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_random_internal_distribution_test_util_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_random_internal_distribution_test_util_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_random_internal_distribution_test_util_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_random_internal_distribution_test_util_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_random_internal_distribution_test_util_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_random_internal_distribution_test_util_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_random_internal_distribution_test_util_TARGET_PROPERTIES TRUE)

########## COMPONENT random_internal_platform TARGET PROPERTIES ######################################

set_property(TARGET absl::random_internal_platform PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_random_internal_platform_LINK_LIBS_DEBUG}
                ${absl_random_internal_platform_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_random_internal_platform_LINK_LIBS_RELEASE}
                ${absl_random_internal_platform_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_random_internal_platform_LINK_LIBS_RELWITHDEBINFO}
                ${absl_random_internal_platform_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_random_internal_platform_LINK_LIBS_MINSIZEREL}
                ${absl_random_internal_platform_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::random_internal_platform PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_random_internal_platform_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_random_internal_platform_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_random_internal_platform_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_random_internal_platform_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::random_internal_platform PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_random_internal_platform_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_random_internal_platform_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_random_internal_platform_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_random_internal_platform_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::random_internal_platform PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_random_internal_platform_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_random_internal_platform_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_random_internal_platform_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_random_internal_platform_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_random_internal_platform_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_random_internal_platform_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_random_internal_platform_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_random_internal_platform_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_random_internal_platform_TARGET_PROPERTIES TRUE)

########## COMPONENT random_internal_randen_hwaes_impl TARGET PROPERTIES ######################################

set_property(TARGET absl::random_internal_randen_hwaes_impl PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_random_internal_randen_hwaes_impl_LINK_LIBS_DEBUG}
                ${absl_random_internal_randen_hwaes_impl_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_random_internal_randen_hwaes_impl_LINK_LIBS_RELEASE}
                ${absl_random_internal_randen_hwaes_impl_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_random_internal_randen_hwaes_impl_LINK_LIBS_RELWITHDEBINFO}
                ${absl_random_internal_randen_hwaes_impl_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_random_internal_randen_hwaes_impl_LINK_LIBS_MINSIZEREL}
                ${absl_random_internal_randen_hwaes_impl_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::random_internal_randen_hwaes_impl PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_random_internal_randen_hwaes_impl_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_random_internal_randen_hwaes_impl_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_random_internal_randen_hwaes_impl_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_random_internal_randen_hwaes_impl_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::random_internal_randen_hwaes_impl PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_random_internal_randen_hwaes_impl_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_random_internal_randen_hwaes_impl_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_random_internal_randen_hwaes_impl_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_random_internal_randen_hwaes_impl_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::random_internal_randen_hwaes_impl PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_random_internal_randen_hwaes_impl_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_random_internal_randen_hwaes_impl_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_random_internal_randen_hwaes_impl_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_random_internal_randen_hwaes_impl_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_random_internal_randen_hwaes_impl_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_random_internal_randen_hwaes_impl_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_random_internal_randen_hwaes_impl_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_random_internal_randen_hwaes_impl_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_random_internal_randen_hwaes_impl_TARGET_PROPERTIES TRUE)

########## COMPONENT random_internal_randen_hwaes TARGET PROPERTIES ######################################

set_property(TARGET absl::random_internal_randen_hwaes PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_random_internal_randen_hwaes_LINK_LIBS_DEBUG}
                ${absl_random_internal_randen_hwaes_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_random_internal_randen_hwaes_LINK_LIBS_RELEASE}
                ${absl_random_internal_randen_hwaes_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_random_internal_randen_hwaes_LINK_LIBS_RELWITHDEBINFO}
                ${absl_random_internal_randen_hwaes_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_random_internal_randen_hwaes_LINK_LIBS_MINSIZEREL}
                ${absl_random_internal_randen_hwaes_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::random_internal_randen_hwaes PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_random_internal_randen_hwaes_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_random_internal_randen_hwaes_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_random_internal_randen_hwaes_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_random_internal_randen_hwaes_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::random_internal_randen_hwaes PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_random_internal_randen_hwaes_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_random_internal_randen_hwaes_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_random_internal_randen_hwaes_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_random_internal_randen_hwaes_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::random_internal_randen_hwaes PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_random_internal_randen_hwaes_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_random_internal_randen_hwaes_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_random_internal_randen_hwaes_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_random_internal_randen_hwaes_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_random_internal_randen_hwaes_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_random_internal_randen_hwaes_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_random_internal_randen_hwaes_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_random_internal_randen_hwaes_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_random_internal_randen_hwaes_TARGET_PROPERTIES TRUE)

########## COMPONENT random_internal_randen_slow TARGET PROPERTIES ######################################

set_property(TARGET absl::random_internal_randen_slow PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_random_internal_randen_slow_LINK_LIBS_DEBUG}
                ${absl_random_internal_randen_slow_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_random_internal_randen_slow_LINK_LIBS_RELEASE}
                ${absl_random_internal_randen_slow_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_random_internal_randen_slow_LINK_LIBS_RELWITHDEBINFO}
                ${absl_random_internal_randen_slow_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_random_internal_randen_slow_LINK_LIBS_MINSIZEREL}
                ${absl_random_internal_randen_slow_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::random_internal_randen_slow PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_random_internal_randen_slow_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_random_internal_randen_slow_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_random_internal_randen_slow_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_random_internal_randen_slow_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::random_internal_randen_slow PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_random_internal_randen_slow_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_random_internal_randen_slow_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_random_internal_randen_slow_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_random_internal_randen_slow_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::random_internal_randen_slow PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_random_internal_randen_slow_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_random_internal_randen_slow_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_random_internal_randen_slow_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_random_internal_randen_slow_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_random_internal_randen_slow_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_random_internal_randen_slow_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_random_internal_randen_slow_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_random_internal_randen_slow_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_random_internal_randen_slow_TARGET_PROPERTIES TRUE)

########## COMPONENT random_internal_randen TARGET PROPERTIES ######################################

set_property(TARGET absl::random_internal_randen PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_random_internal_randen_LINK_LIBS_DEBUG}
                ${absl_random_internal_randen_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_random_internal_randen_LINK_LIBS_RELEASE}
                ${absl_random_internal_randen_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_random_internal_randen_LINK_LIBS_RELWITHDEBINFO}
                ${absl_random_internal_randen_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_random_internal_randen_LINK_LIBS_MINSIZEREL}
                ${absl_random_internal_randen_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::random_internal_randen PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_random_internal_randen_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_random_internal_randen_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_random_internal_randen_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_random_internal_randen_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::random_internal_randen PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_random_internal_randen_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_random_internal_randen_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_random_internal_randen_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_random_internal_randen_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::random_internal_randen PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_random_internal_randen_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_random_internal_randen_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_random_internal_randen_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_random_internal_randen_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_random_internal_randen_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_random_internal_randen_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_random_internal_randen_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_random_internal_randen_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_random_internal_randen_TARGET_PROPERTIES TRUE)

########## COMPONENT random_internal_iostream_state_saver TARGET PROPERTIES ######################################

set_property(TARGET absl::random_internal_iostream_state_saver PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_random_internal_iostream_state_saver_LINK_LIBS_DEBUG}
                ${absl_random_internal_iostream_state_saver_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_random_internal_iostream_state_saver_LINK_LIBS_RELEASE}
                ${absl_random_internal_iostream_state_saver_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_random_internal_iostream_state_saver_LINK_LIBS_RELWITHDEBINFO}
                ${absl_random_internal_iostream_state_saver_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_random_internal_iostream_state_saver_LINK_LIBS_MINSIZEREL}
                ${absl_random_internal_iostream_state_saver_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::random_internal_iostream_state_saver PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_random_internal_iostream_state_saver_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_random_internal_iostream_state_saver_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_random_internal_iostream_state_saver_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_random_internal_iostream_state_saver_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::random_internal_iostream_state_saver PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_random_internal_iostream_state_saver_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_random_internal_iostream_state_saver_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_random_internal_iostream_state_saver_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_random_internal_iostream_state_saver_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::random_internal_iostream_state_saver PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_random_internal_iostream_state_saver_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_random_internal_iostream_state_saver_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_random_internal_iostream_state_saver_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_random_internal_iostream_state_saver_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_random_internal_iostream_state_saver_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_random_internal_iostream_state_saver_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_random_internal_iostream_state_saver_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_random_internal_iostream_state_saver_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_random_internal_iostream_state_saver_TARGET_PROPERTIES TRUE)

########## COMPONENT random_internal_randen_engine TARGET PROPERTIES ######################################

set_property(TARGET absl::random_internal_randen_engine PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_random_internal_randen_engine_LINK_LIBS_DEBUG}
                ${absl_random_internal_randen_engine_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_random_internal_randen_engine_LINK_LIBS_RELEASE}
                ${absl_random_internal_randen_engine_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_random_internal_randen_engine_LINK_LIBS_RELWITHDEBINFO}
                ${absl_random_internal_randen_engine_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_random_internal_randen_engine_LINK_LIBS_MINSIZEREL}
                ${absl_random_internal_randen_engine_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::random_internal_randen_engine PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_random_internal_randen_engine_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_random_internal_randen_engine_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_random_internal_randen_engine_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_random_internal_randen_engine_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::random_internal_randen_engine PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_random_internal_randen_engine_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_random_internal_randen_engine_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_random_internal_randen_engine_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_random_internal_randen_engine_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::random_internal_randen_engine PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_random_internal_randen_engine_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_random_internal_randen_engine_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_random_internal_randen_engine_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_random_internal_randen_engine_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_random_internal_randen_engine_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_random_internal_randen_engine_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_random_internal_randen_engine_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_random_internal_randen_engine_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_random_internal_randen_engine_TARGET_PROPERTIES TRUE)

########## COMPONENT random_internal_fastmath TARGET PROPERTIES ######################################

set_property(TARGET absl::random_internal_fastmath PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_random_internal_fastmath_LINK_LIBS_DEBUG}
                ${absl_random_internal_fastmath_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_random_internal_fastmath_LINK_LIBS_RELEASE}
                ${absl_random_internal_fastmath_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_random_internal_fastmath_LINK_LIBS_RELWITHDEBINFO}
                ${absl_random_internal_fastmath_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_random_internal_fastmath_LINK_LIBS_MINSIZEREL}
                ${absl_random_internal_fastmath_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::random_internal_fastmath PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_random_internal_fastmath_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_random_internal_fastmath_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_random_internal_fastmath_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_random_internal_fastmath_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::random_internal_fastmath PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_random_internal_fastmath_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_random_internal_fastmath_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_random_internal_fastmath_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_random_internal_fastmath_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::random_internal_fastmath PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_random_internal_fastmath_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_random_internal_fastmath_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_random_internal_fastmath_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_random_internal_fastmath_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_random_internal_fastmath_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_random_internal_fastmath_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_random_internal_fastmath_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_random_internal_fastmath_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_random_internal_fastmath_TARGET_PROPERTIES TRUE)

########## COMPONENT random_internal_pcg_engine TARGET PROPERTIES ######################################

set_property(TARGET absl::random_internal_pcg_engine PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_random_internal_pcg_engine_LINK_LIBS_DEBUG}
                ${absl_random_internal_pcg_engine_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_random_internal_pcg_engine_LINK_LIBS_RELEASE}
                ${absl_random_internal_pcg_engine_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_random_internal_pcg_engine_LINK_LIBS_RELWITHDEBINFO}
                ${absl_random_internal_pcg_engine_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_random_internal_pcg_engine_LINK_LIBS_MINSIZEREL}
                ${absl_random_internal_pcg_engine_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::random_internal_pcg_engine PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_random_internal_pcg_engine_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_random_internal_pcg_engine_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_random_internal_pcg_engine_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_random_internal_pcg_engine_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::random_internal_pcg_engine PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_random_internal_pcg_engine_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_random_internal_pcg_engine_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_random_internal_pcg_engine_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_random_internal_pcg_engine_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::random_internal_pcg_engine PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_random_internal_pcg_engine_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_random_internal_pcg_engine_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_random_internal_pcg_engine_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_random_internal_pcg_engine_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_random_internal_pcg_engine_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_random_internal_pcg_engine_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_random_internal_pcg_engine_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_random_internal_pcg_engine_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_random_internal_pcg_engine_TARGET_PROPERTIES TRUE)

########## COMPONENT random_internal_fast_uniform_bits TARGET PROPERTIES ######################################

set_property(TARGET absl::random_internal_fast_uniform_bits PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_random_internal_fast_uniform_bits_LINK_LIBS_DEBUG}
                ${absl_random_internal_fast_uniform_bits_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_random_internal_fast_uniform_bits_LINK_LIBS_RELEASE}
                ${absl_random_internal_fast_uniform_bits_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_random_internal_fast_uniform_bits_LINK_LIBS_RELWITHDEBINFO}
                ${absl_random_internal_fast_uniform_bits_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_random_internal_fast_uniform_bits_LINK_LIBS_MINSIZEREL}
                ${absl_random_internal_fast_uniform_bits_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::random_internal_fast_uniform_bits PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_random_internal_fast_uniform_bits_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_random_internal_fast_uniform_bits_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_random_internal_fast_uniform_bits_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_random_internal_fast_uniform_bits_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::random_internal_fast_uniform_bits PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_random_internal_fast_uniform_bits_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_random_internal_fast_uniform_bits_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_random_internal_fast_uniform_bits_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_random_internal_fast_uniform_bits_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::random_internal_fast_uniform_bits PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_random_internal_fast_uniform_bits_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_random_internal_fast_uniform_bits_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_random_internal_fast_uniform_bits_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_random_internal_fast_uniform_bits_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_random_internal_fast_uniform_bits_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_random_internal_fast_uniform_bits_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_random_internal_fast_uniform_bits_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_random_internal_fast_uniform_bits_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_random_internal_fast_uniform_bits_TARGET_PROPERTIES TRUE)

########## COMPONENT random_internal_seed_material TARGET PROPERTIES ######################################

set_property(TARGET absl::random_internal_seed_material PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_random_internal_seed_material_LINK_LIBS_DEBUG}
                ${absl_random_internal_seed_material_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_random_internal_seed_material_LINK_LIBS_RELEASE}
                ${absl_random_internal_seed_material_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_random_internal_seed_material_LINK_LIBS_RELWITHDEBINFO}
                ${absl_random_internal_seed_material_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_random_internal_seed_material_LINK_LIBS_MINSIZEREL}
                ${absl_random_internal_seed_material_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::random_internal_seed_material PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_random_internal_seed_material_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_random_internal_seed_material_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_random_internal_seed_material_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_random_internal_seed_material_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::random_internal_seed_material PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_random_internal_seed_material_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_random_internal_seed_material_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_random_internal_seed_material_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_random_internal_seed_material_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::random_internal_seed_material PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_random_internal_seed_material_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_random_internal_seed_material_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_random_internal_seed_material_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_random_internal_seed_material_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_random_internal_seed_material_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_random_internal_seed_material_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_random_internal_seed_material_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_random_internal_seed_material_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_random_internal_seed_material_TARGET_PROPERTIES TRUE)

########## COMPONENT random_internal_salted_seed_seq TARGET PROPERTIES ######################################

set_property(TARGET absl::random_internal_salted_seed_seq PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_random_internal_salted_seed_seq_LINK_LIBS_DEBUG}
                ${absl_random_internal_salted_seed_seq_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_random_internal_salted_seed_seq_LINK_LIBS_RELEASE}
                ${absl_random_internal_salted_seed_seq_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_random_internal_salted_seed_seq_LINK_LIBS_RELWITHDEBINFO}
                ${absl_random_internal_salted_seed_seq_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_random_internal_salted_seed_seq_LINK_LIBS_MINSIZEREL}
                ${absl_random_internal_salted_seed_seq_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::random_internal_salted_seed_seq PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_random_internal_salted_seed_seq_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_random_internal_salted_seed_seq_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_random_internal_salted_seed_seq_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_random_internal_salted_seed_seq_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::random_internal_salted_seed_seq PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_random_internal_salted_seed_seq_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_random_internal_salted_seed_seq_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_random_internal_salted_seed_seq_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_random_internal_salted_seed_seq_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::random_internal_salted_seed_seq PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_random_internal_salted_seed_seq_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_random_internal_salted_seed_seq_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_random_internal_salted_seed_seq_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_random_internal_salted_seed_seq_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_random_internal_salted_seed_seq_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_random_internal_salted_seed_seq_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_random_internal_salted_seed_seq_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_random_internal_salted_seed_seq_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_random_internal_salted_seed_seq_TARGET_PROPERTIES TRUE)

########## COMPONENT random_seed_gen_exception TARGET PROPERTIES ######################################

set_property(TARGET absl::random_seed_gen_exception PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_random_seed_gen_exception_LINK_LIBS_DEBUG}
                ${absl_random_seed_gen_exception_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_random_seed_gen_exception_LINK_LIBS_RELEASE}
                ${absl_random_seed_gen_exception_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_random_seed_gen_exception_LINK_LIBS_RELWITHDEBINFO}
                ${absl_random_seed_gen_exception_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_random_seed_gen_exception_LINK_LIBS_MINSIZEREL}
                ${absl_random_seed_gen_exception_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::random_seed_gen_exception PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_random_seed_gen_exception_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_random_seed_gen_exception_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_random_seed_gen_exception_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_random_seed_gen_exception_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::random_seed_gen_exception PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_random_seed_gen_exception_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_random_seed_gen_exception_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_random_seed_gen_exception_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_random_seed_gen_exception_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::random_seed_gen_exception PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_random_seed_gen_exception_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_random_seed_gen_exception_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_random_seed_gen_exception_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_random_seed_gen_exception_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_random_seed_gen_exception_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_random_seed_gen_exception_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_random_seed_gen_exception_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_random_seed_gen_exception_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_random_seed_gen_exception_TARGET_PROPERTIES TRUE)

########## COMPONENT random_internal_pool_urbg TARGET PROPERTIES ######################################

set_property(TARGET absl::random_internal_pool_urbg PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_random_internal_pool_urbg_LINK_LIBS_DEBUG}
                ${absl_random_internal_pool_urbg_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_random_internal_pool_urbg_LINK_LIBS_RELEASE}
                ${absl_random_internal_pool_urbg_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_random_internal_pool_urbg_LINK_LIBS_RELWITHDEBINFO}
                ${absl_random_internal_pool_urbg_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_random_internal_pool_urbg_LINK_LIBS_MINSIZEREL}
                ${absl_random_internal_pool_urbg_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::random_internal_pool_urbg PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_random_internal_pool_urbg_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_random_internal_pool_urbg_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_random_internal_pool_urbg_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_random_internal_pool_urbg_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::random_internal_pool_urbg PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_random_internal_pool_urbg_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_random_internal_pool_urbg_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_random_internal_pool_urbg_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_random_internal_pool_urbg_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::random_internal_pool_urbg PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_random_internal_pool_urbg_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_random_internal_pool_urbg_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_random_internal_pool_urbg_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_random_internal_pool_urbg_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_random_internal_pool_urbg_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_random_internal_pool_urbg_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_random_internal_pool_urbg_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_random_internal_pool_urbg_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_random_internal_pool_urbg_TARGET_PROPERTIES TRUE)

########## COMPONENT random_internal_nonsecure_base TARGET PROPERTIES ######################################

set_property(TARGET absl::random_internal_nonsecure_base PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_random_internal_nonsecure_base_LINK_LIBS_DEBUG}
                ${absl_random_internal_nonsecure_base_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_random_internal_nonsecure_base_LINK_LIBS_RELEASE}
                ${absl_random_internal_nonsecure_base_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_random_internal_nonsecure_base_LINK_LIBS_RELWITHDEBINFO}
                ${absl_random_internal_nonsecure_base_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_random_internal_nonsecure_base_LINK_LIBS_MINSIZEREL}
                ${absl_random_internal_nonsecure_base_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::random_internal_nonsecure_base PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_random_internal_nonsecure_base_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_random_internal_nonsecure_base_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_random_internal_nonsecure_base_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_random_internal_nonsecure_base_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::random_internal_nonsecure_base PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_random_internal_nonsecure_base_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_random_internal_nonsecure_base_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_random_internal_nonsecure_base_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_random_internal_nonsecure_base_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::random_internal_nonsecure_base PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_random_internal_nonsecure_base_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_random_internal_nonsecure_base_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_random_internal_nonsecure_base_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_random_internal_nonsecure_base_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_random_internal_nonsecure_base_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_random_internal_nonsecure_base_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_random_internal_nonsecure_base_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_random_internal_nonsecure_base_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_random_internal_nonsecure_base_TARGET_PROPERTIES TRUE)

########## COMPONENT random_internal_wide_multiply TARGET PROPERTIES ######################################

set_property(TARGET absl::random_internal_wide_multiply PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_random_internal_wide_multiply_LINK_LIBS_DEBUG}
                ${absl_random_internal_wide_multiply_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_random_internal_wide_multiply_LINK_LIBS_RELEASE}
                ${absl_random_internal_wide_multiply_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_random_internal_wide_multiply_LINK_LIBS_RELWITHDEBINFO}
                ${absl_random_internal_wide_multiply_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_random_internal_wide_multiply_LINK_LIBS_MINSIZEREL}
                ${absl_random_internal_wide_multiply_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::random_internal_wide_multiply PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_random_internal_wide_multiply_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_random_internal_wide_multiply_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_random_internal_wide_multiply_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_random_internal_wide_multiply_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::random_internal_wide_multiply PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_random_internal_wide_multiply_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_random_internal_wide_multiply_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_random_internal_wide_multiply_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_random_internal_wide_multiply_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::random_internal_wide_multiply PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_random_internal_wide_multiply_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_random_internal_wide_multiply_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_random_internal_wide_multiply_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_random_internal_wide_multiply_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_random_internal_wide_multiply_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_random_internal_wide_multiply_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_random_internal_wide_multiply_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_random_internal_wide_multiply_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_random_internal_wide_multiply_TARGET_PROPERTIES TRUE)

########## COMPONENT random_internal_generate_real TARGET PROPERTIES ######################################

set_property(TARGET absl::random_internal_generate_real PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_random_internal_generate_real_LINK_LIBS_DEBUG}
                ${absl_random_internal_generate_real_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_random_internal_generate_real_LINK_LIBS_RELEASE}
                ${absl_random_internal_generate_real_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_random_internal_generate_real_LINK_LIBS_RELWITHDEBINFO}
                ${absl_random_internal_generate_real_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_random_internal_generate_real_LINK_LIBS_MINSIZEREL}
                ${absl_random_internal_generate_real_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::random_internal_generate_real PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_random_internal_generate_real_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_random_internal_generate_real_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_random_internal_generate_real_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_random_internal_generate_real_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::random_internal_generate_real PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_random_internal_generate_real_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_random_internal_generate_real_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_random_internal_generate_real_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_random_internal_generate_real_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::random_internal_generate_real PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_random_internal_generate_real_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_random_internal_generate_real_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_random_internal_generate_real_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_random_internal_generate_real_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_random_internal_generate_real_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_random_internal_generate_real_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_random_internal_generate_real_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_random_internal_generate_real_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_random_internal_generate_real_TARGET_PROPERTIES TRUE)

########## COMPONENT random_internal_distribution_caller TARGET PROPERTIES ######################################

set_property(TARGET absl::random_internal_distribution_caller PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_random_internal_distribution_caller_LINK_LIBS_DEBUG}
                ${absl_random_internal_distribution_caller_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_random_internal_distribution_caller_LINK_LIBS_RELEASE}
                ${absl_random_internal_distribution_caller_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_random_internal_distribution_caller_LINK_LIBS_RELWITHDEBINFO}
                ${absl_random_internal_distribution_caller_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_random_internal_distribution_caller_LINK_LIBS_MINSIZEREL}
                ${absl_random_internal_distribution_caller_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::random_internal_distribution_caller PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_random_internal_distribution_caller_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_random_internal_distribution_caller_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_random_internal_distribution_caller_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_random_internal_distribution_caller_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::random_internal_distribution_caller PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_random_internal_distribution_caller_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_random_internal_distribution_caller_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_random_internal_distribution_caller_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_random_internal_distribution_caller_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::random_internal_distribution_caller PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_random_internal_distribution_caller_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_random_internal_distribution_caller_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_random_internal_distribution_caller_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_random_internal_distribution_caller_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_random_internal_distribution_caller_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_random_internal_distribution_caller_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_random_internal_distribution_caller_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_random_internal_distribution_caller_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_random_internal_distribution_caller_TARGET_PROPERTIES TRUE)

########## COMPONENT random_seed_sequences TARGET PROPERTIES ######################################

set_property(TARGET absl::random_seed_sequences PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_random_seed_sequences_LINK_LIBS_DEBUG}
                ${absl_random_seed_sequences_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_random_seed_sequences_LINK_LIBS_RELEASE}
                ${absl_random_seed_sequences_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_random_seed_sequences_LINK_LIBS_RELWITHDEBINFO}
                ${absl_random_seed_sequences_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_random_seed_sequences_LINK_LIBS_MINSIZEREL}
                ${absl_random_seed_sequences_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::random_seed_sequences PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_random_seed_sequences_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_random_seed_sequences_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_random_seed_sequences_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_random_seed_sequences_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::random_seed_sequences PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_random_seed_sequences_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_random_seed_sequences_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_random_seed_sequences_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_random_seed_sequences_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::random_seed_sequences PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_random_seed_sequences_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_random_seed_sequences_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_random_seed_sequences_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_random_seed_sequences_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_random_seed_sequences_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_random_seed_sequences_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_random_seed_sequences_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_random_seed_sequences_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_random_seed_sequences_TARGET_PROPERTIES TRUE)

########## COMPONENT random_distributions TARGET PROPERTIES ######################################

set_property(TARGET absl::random_distributions PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_random_distributions_LINK_LIBS_DEBUG}
                ${absl_random_distributions_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_random_distributions_LINK_LIBS_RELEASE}
                ${absl_random_distributions_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_random_distributions_LINK_LIBS_RELWITHDEBINFO}
                ${absl_random_distributions_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_random_distributions_LINK_LIBS_MINSIZEREL}
                ${absl_random_distributions_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::random_distributions PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_random_distributions_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_random_distributions_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_random_distributions_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_random_distributions_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::random_distributions PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_random_distributions_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_random_distributions_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_random_distributions_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_random_distributions_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::random_distributions PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_random_distributions_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_random_distributions_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_random_distributions_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_random_distributions_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_random_distributions_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_random_distributions_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_random_distributions_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_random_distributions_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_random_distributions_TARGET_PROPERTIES TRUE)

########## COMPONENT random_internal_mock_helpers TARGET PROPERTIES ######################################

set_property(TARGET absl::random_internal_mock_helpers PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_random_internal_mock_helpers_LINK_LIBS_DEBUG}
                ${absl_random_internal_mock_helpers_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_random_internal_mock_helpers_LINK_LIBS_RELEASE}
                ${absl_random_internal_mock_helpers_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_random_internal_mock_helpers_LINK_LIBS_RELWITHDEBINFO}
                ${absl_random_internal_mock_helpers_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_random_internal_mock_helpers_LINK_LIBS_MINSIZEREL}
                ${absl_random_internal_mock_helpers_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::random_internal_mock_helpers PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_random_internal_mock_helpers_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_random_internal_mock_helpers_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_random_internal_mock_helpers_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_random_internal_mock_helpers_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::random_internal_mock_helpers PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_random_internal_mock_helpers_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_random_internal_mock_helpers_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_random_internal_mock_helpers_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_random_internal_mock_helpers_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::random_internal_mock_helpers PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_random_internal_mock_helpers_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_random_internal_mock_helpers_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_random_internal_mock_helpers_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_random_internal_mock_helpers_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_random_internal_mock_helpers_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_random_internal_mock_helpers_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_random_internal_mock_helpers_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_random_internal_mock_helpers_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_random_internal_mock_helpers_TARGET_PROPERTIES TRUE)

########## COMPONENT random_bit_gen_ref TARGET PROPERTIES ######################################

set_property(TARGET absl::random_bit_gen_ref PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_random_bit_gen_ref_LINK_LIBS_DEBUG}
                ${absl_random_bit_gen_ref_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_random_bit_gen_ref_LINK_LIBS_RELEASE}
                ${absl_random_bit_gen_ref_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_random_bit_gen_ref_LINK_LIBS_RELWITHDEBINFO}
                ${absl_random_bit_gen_ref_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_random_bit_gen_ref_LINK_LIBS_MINSIZEREL}
                ${absl_random_bit_gen_ref_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::random_bit_gen_ref PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_random_bit_gen_ref_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_random_bit_gen_ref_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_random_bit_gen_ref_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_random_bit_gen_ref_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::random_bit_gen_ref PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_random_bit_gen_ref_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_random_bit_gen_ref_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_random_bit_gen_ref_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_random_bit_gen_ref_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::random_bit_gen_ref PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_random_bit_gen_ref_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_random_bit_gen_ref_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_random_bit_gen_ref_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_random_bit_gen_ref_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_random_bit_gen_ref_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_random_bit_gen_ref_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_random_bit_gen_ref_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_random_bit_gen_ref_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_random_bit_gen_ref_TARGET_PROPERTIES TRUE)

########## COMPONENT random_random TARGET PROPERTIES ######################################

set_property(TARGET absl::random_random PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_random_random_LINK_LIBS_DEBUG}
                ${absl_random_random_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_random_random_LINK_LIBS_RELEASE}
                ${absl_random_random_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_random_random_LINK_LIBS_RELWITHDEBINFO}
                ${absl_random_random_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_random_random_LINK_LIBS_MINSIZEREL}
                ${absl_random_random_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::random_random PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_random_random_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_random_random_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_random_random_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_random_random_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::random_random PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_random_random_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_random_random_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_random_random_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_random_random_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::random_random PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_random_random_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_random_random_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_random_random_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_random_random_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_random_random_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_random_random_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_random_random_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_random_random_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_random_random_TARGET_PROPERTIES TRUE)

########## COMPONENT periodic_sampler TARGET PROPERTIES ######################################

set_property(TARGET absl::periodic_sampler PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_periodic_sampler_LINK_LIBS_DEBUG}
                ${absl_periodic_sampler_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_periodic_sampler_LINK_LIBS_RELEASE}
                ${absl_periodic_sampler_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_periodic_sampler_LINK_LIBS_RELWITHDEBINFO}
                ${absl_periodic_sampler_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_periodic_sampler_LINK_LIBS_MINSIZEREL}
                ${absl_periodic_sampler_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::periodic_sampler PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_periodic_sampler_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_periodic_sampler_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_periodic_sampler_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_periodic_sampler_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::periodic_sampler PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_periodic_sampler_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_periodic_sampler_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_periodic_sampler_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_periodic_sampler_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::periodic_sampler PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_periodic_sampler_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_periodic_sampler_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_periodic_sampler_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_periodic_sampler_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_periodic_sampler_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_periodic_sampler_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_periodic_sampler_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_periodic_sampler_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_periodic_sampler_TARGET_PROPERTIES TRUE)

########## COMPONENT sample_recorder TARGET PROPERTIES ######################################

set_property(TARGET absl::sample_recorder PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_sample_recorder_LINK_LIBS_DEBUG}
                ${absl_sample_recorder_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_sample_recorder_LINK_LIBS_RELEASE}
                ${absl_sample_recorder_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_sample_recorder_LINK_LIBS_RELWITHDEBINFO}
                ${absl_sample_recorder_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_sample_recorder_LINK_LIBS_MINSIZEREL}
                ${absl_sample_recorder_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::sample_recorder PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_sample_recorder_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_sample_recorder_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_sample_recorder_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_sample_recorder_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::sample_recorder PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_sample_recorder_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_sample_recorder_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_sample_recorder_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_sample_recorder_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::sample_recorder PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_sample_recorder_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_sample_recorder_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_sample_recorder_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_sample_recorder_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_sample_recorder_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_sample_recorder_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_sample_recorder_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_sample_recorder_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_sample_recorder_TARGET_PROPERTIES TRUE)

########## COMPONENT numeric TARGET PROPERTIES ######################################

set_property(TARGET absl::numeric PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_numeric_LINK_LIBS_DEBUG}
                ${absl_numeric_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_numeric_LINK_LIBS_RELEASE}
                ${absl_numeric_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_numeric_LINK_LIBS_RELWITHDEBINFO}
                ${absl_numeric_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_numeric_LINK_LIBS_MINSIZEREL}
                ${absl_numeric_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::numeric PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_numeric_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_numeric_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_numeric_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_numeric_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::numeric PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_numeric_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_numeric_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_numeric_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_numeric_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::numeric PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_numeric_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_numeric_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_numeric_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_numeric_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_numeric_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_numeric_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_numeric_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_numeric_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_numeric_TARGET_PROPERTIES TRUE)

########## COMPONENT log_internal_config TARGET PROPERTIES ######################################

set_property(TARGET absl::log_internal_config PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_log_internal_config_LINK_LIBS_DEBUG}
                ${absl_log_internal_config_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_log_internal_config_LINK_LIBS_RELEASE}
                ${absl_log_internal_config_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_log_internal_config_LINK_LIBS_RELWITHDEBINFO}
                ${absl_log_internal_config_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_log_internal_config_LINK_LIBS_MINSIZEREL}
                ${absl_log_internal_config_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::log_internal_config PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_log_internal_config_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_log_internal_config_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_log_internal_config_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_log_internal_config_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::log_internal_config PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_log_internal_config_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_log_internal_config_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_log_internal_config_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_log_internal_config_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::log_internal_config PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_log_internal_config_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_log_internal_config_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_log_internal_config_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_log_internal_config_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_log_internal_config_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_log_internal_config_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_log_internal_config_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_log_internal_config_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_log_internal_config_TARGET_PROPERTIES TRUE)

########## COMPONENT log_entry TARGET PROPERTIES ######################################

set_property(TARGET absl::log_entry PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_log_entry_LINK_LIBS_DEBUG}
                ${absl_log_entry_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_log_entry_LINK_LIBS_RELEASE}
                ${absl_log_entry_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_log_entry_LINK_LIBS_RELWITHDEBINFO}
                ${absl_log_entry_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_log_entry_LINK_LIBS_MINSIZEREL}
                ${absl_log_entry_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::log_entry PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_log_entry_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_log_entry_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_log_entry_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_log_entry_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::log_entry PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_log_entry_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_log_entry_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_log_entry_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_log_entry_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::log_entry PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_log_entry_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_log_entry_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_log_entry_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_log_entry_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_log_entry_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_log_entry_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_log_entry_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_log_entry_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_log_entry_TARGET_PROPERTIES TRUE)

########## COMPONENT log_sink TARGET PROPERTIES ######################################

set_property(TARGET absl::log_sink PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_log_sink_LINK_LIBS_DEBUG}
                ${absl_log_sink_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_log_sink_LINK_LIBS_RELEASE}
                ${absl_log_sink_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_log_sink_LINK_LIBS_RELWITHDEBINFO}
                ${absl_log_sink_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_log_sink_LINK_LIBS_MINSIZEREL}
                ${absl_log_sink_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::log_sink PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_log_sink_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_log_sink_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_log_sink_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_log_sink_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::log_sink PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_log_sink_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_log_sink_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_log_sink_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_log_sink_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::log_sink PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_log_sink_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_log_sink_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_log_sink_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_log_sink_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_log_sink_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_log_sink_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_log_sink_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_log_sink_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_log_sink_TARGET_PROPERTIES TRUE)

########## COMPONENT low_level_hash TARGET PROPERTIES ######################################

set_property(TARGET absl::low_level_hash PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_low_level_hash_LINK_LIBS_DEBUG}
                ${absl_low_level_hash_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_low_level_hash_LINK_LIBS_RELEASE}
                ${absl_low_level_hash_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_low_level_hash_LINK_LIBS_RELWITHDEBINFO}
                ${absl_low_level_hash_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_low_level_hash_LINK_LIBS_MINSIZEREL}
                ${absl_low_level_hash_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::low_level_hash PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_low_level_hash_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_low_level_hash_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_low_level_hash_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_low_level_hash_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::low_level_hash PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_low_level_hash_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_low_level_hash_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_low_level_hash_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_low_level_hash_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::low_level_hash PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_low_level_hash_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_low_level_hash_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_low_level_hash_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_low_level_hash_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_low_level_hash_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_low_level_hash_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_low_level_hash_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_low_level_hash_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_low_level_hash_TARGET_PROPERTIES TRUE)

########## COMPONENT city TARGET PROPERTIES ######################################

set_property(TARGET absl::city PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_city_LINK_LIBS_DEBUG}
                ${absl_city_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_city_LINK_LIBS_RELEASE}
                ${absl_city_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_city_LINK_LIBS_RELWITHDEBINFO}
                ${absl_city_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_city_LINK_LIBS_MINSIZEREL}
                ${absl_city_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::city PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_city_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_city_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_city_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_city_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::city PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_city_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_city_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_city_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_city_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::city PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_city_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_city_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_city_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_city_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_city_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_city_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_city_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_city_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_city_TARGET_PROPERTIES TRUE)

########## COMPONENT hash TARGET PROPERTIES ######################################

set_property(TARGET absl::hash PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_hash_LINK_LIBS_DEBUG}
                ${absl_hash_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_hash_LINK_LIBS_RELEASE}
                ${absl_hash_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_hash_LINK_LIBS_RELWITHDEBINFO}
                ${absl_hash_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_hash_LINK_LIBS_MINSIZEREL}
                ${absl_hash_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::hash PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_hash_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_hash_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_hash_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_hash_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::hash PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_hash_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_hash_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_hash_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_hash_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::hash PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_hash_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_hash_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_hash_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_hash_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_hash_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_hash_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_hash_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_hash_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_hash_TARGET_PROPERTIES TRUE)

########## COMPONENT log_globals TARGET PROPERTIES ######################################

set_property(TARGET absl::log_globals PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_log_globals_LINK_LIBS_DEBUG}
                ${absl_log_globals_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_log_globals_LINK_LIBS_RELEASE}
                ${absl_log_globals_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_log_globals_LINK_LIBS_RELWITHDEBINFO}
                ${absl_log_globals_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_log_globals_LINK_LIBS_MINSIZEREL}
                ${absl_log_globals_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::log_globals PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_log_globals_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_log_globals_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_log_globals_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_log_globals_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::log_globals PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_log_globals_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_log_globals_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_log_globals_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_log_globals_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::log_globals PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_log_globals_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_log_globals_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_log_globals_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_log_globals_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_log_globals_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_log_globals_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_log_globals_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_log_globals_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_log_globals_TARGET_PROPERTIES TRUE)

########## COMPONENT log_internal_globals TARGET PROPERTIES ######################################

set_property(TARGET absl::log_internal_globals PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_log_internal_globals_LINK_LIBS_DEBUG}
                ${absl_log_internal_globals_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_log_internal_globals_LINK_LIBS_RELEASE}
                ${absl_log_internal_globals_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_log_internal_globals_LINK_LIBS_RELWITHDEBINFO}
                ${absl_log_internal_globals_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_log_internal_globals_LINK_LIBS_MINSIZEREL}
                ${absl_log_internal_globals_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::log_internal_globals PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_log_internal_globals_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_log_internal_globals_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_log_internal_globals_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_log_internal_globals_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::log_internal_globals PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_log_internal_globals_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_log_internal_globals_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_log_internal_globals_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_log_internal_globals_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::log_internal_globals PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_log_internal_globals_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_log_internal_globals_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_log_internal_globals_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_log_internal_globals_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_log_internal_globals_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_log_internal_globals_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_log_internal_globals_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_log_internal_globals_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_log_internal_globals_TARGET_PROPERTIES TRUE)

########## COMPONENT cleanup_internal TARGET PROPERTIES ######################################

set_property(TARGET absl::cleanup_internal PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_cleanup_internal_LINK_LIBS_DEBUG}
                ${absl_cleanup_internal_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_cleanup_internal_LINK_LIBS_RELEASE}
                ${absl_cleanup_internal_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_cleanup_internal_LINK_LIBS_RELWITHDEBINFO}
                ${absl_cleanup_internal_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_cleanup_internal_LINK_LIBS_MINSIZEREL}
                ${absl_cleanup_internal_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::cleanup_internal PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_cleanup_internal_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_cleanup_internal_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_cleanup_internal_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_cleanup_internal_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::cleanup_internal PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_cleanup_internal_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_cleanup_internal_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_cleanup_internal_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_cleanup_internal_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::cleanup_internal PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_cleanup_internal_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_cleanup_internal_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_cleanup_internal_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_cleanup_internal_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_cleanup_internal_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_cleanup_internal_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_cleanup_internal_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_cleanup_internal_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_cleanup_internal_TARGET_PROPERTIES TRUE)

########## COMPONENT cleanup TARGET PROPERTIES ######################################

set_property(TARGET absl::cleanup PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_cleanup_LINK_LIBS_DEBUG}
                ${absl_cleanup_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_cleanup_LINK_LIBS_RELEASE}
                ${absl_cleanup_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_cleanup_LINK_LIBS_RELWITHDEBINFO}
                ${absl_cleanup_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_cleanup_LINK_LIBS_MINSIZEREL}
                ${absl_cleanup_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::cleanup PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_cleanup_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_cleanup_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_cleanup_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_cleanup_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::cleanup PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_cleanup_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_cleanup_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_cleanup_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_cleanup_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::cleanup PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_cleanup_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_cleanup_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_cleanup_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_cleanup_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_cleanup_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_cleanup_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_cleanup_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_cleanup_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_cleanup_TARGET_PROPERTIES TRUE)

########## COMPONENT log_internal_log_sink_set TARGET PROPERTIES ######################################

set_property(TARGET absl::log_internal_log_sink_set PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_log_internal_log_sink_set_LINK_LIBS_DEBUG}
                ${absl_log_internal_log_sink_set_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_log_internal_log_sink_set_LINK_LIBS_RELEASE}
                ${absl_log_internal_log_sink_set_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_log_internal_log_sink_set_LINK_LIBS_RELWITHDEBINFO}
                ${absl_log_internal_log_sink_set_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_log_internal_log_sink_set_LINK_LIBS_MINSIZEREL}
                ${absl_log_internal_log_sink_set_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::log_internal_log_sink_set PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_log_internal_log_sink_set_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_log_internal_log_sink_set_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_log_internal_log_sink_set_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_log_internal_log_sink_set_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::log_internal_log_sink_set PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_log_internal_log_sink_set_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_log_internal_log_sink_set_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_log_internal_log_sink_set_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_log_internal_log_sink_set_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::log_internal_log_sink_set PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_log_internal_log_sink_set_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_log_internal_log_sink_set_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_log_internal_log_sink_set_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_log_internal_log_sink_set_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_log_internal_log_sink_set_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_log_internal_log_sink_set_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_log_internal_log_sink_set_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_log_internal_log_sink_set_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_log_internal_log_sink_set_TARGET_PROPERTIES TRUE)

########## COMPONENT log_sink_registry TARGET PROPERTIES ######################################

set_property(TARGET absl::log_sink_registry PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_log_sink_registry_LINK_LIBS_DEBUG}
                ${absl_log_sink_registry_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_log_sink_registry_LINK_LIBS_RELEASE}
                ${absl_log_sink_registry_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_log_sink_registry_LINK_LIBS_RELWITHDEBINFO}
                ${absl_log_sink_registry_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_log_sink_registry_LINK_LIBS_MINSIZEREL}
                ${absl_log_sink_registry_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::log_sink_registry PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_log_sink_registry_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_log_sink_registry_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_log_sink_registry_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_log_sink_registry_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::log_sink_registry PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_log_sink_registry_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_log_sink_registry_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_log_sink_registry_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_log_sink_registry_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::log_sink_registry PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_log_sink_registry_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_log_sink_registry_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_log_sink_registry_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_log_sink_registry_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_log_sink_registry_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_log_sink_registry_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_log_sink_registry_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_log_sink_registry_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_log_sink_registry_TARGET_PROPERTIES TRUE)

########## COMPONENT log_internal_append_truncated TARGET PROPERTIES ######################################

set_property(TARGET absl::log_internal_append_truncated PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_log_internal_append_truncated_LINK_LIBS_DEBUG}
                ${absl_log_internal_append_truncated_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_log_internal_append_truncated_LINK_LIBS_RELEASE}
                ${absl_log_internal_append_truncated_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_log_internal_append_truncated_LINK_LIBS_RELWITHDEBINFO}
                ${absl_log_internal_append_truncated_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_log_internal_append_truncated_LINK_LIBS_MINSIZEREL}
                ${absl_log_internal_append_truncated_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::log_internal_append_truncated PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_log_internal_append_truncated_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_log_internal_append_truncated_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_log_internal_append_truncated_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_log_internal_append_truncated_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::log_internal_append_truncated PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_log_internal_append_truncated_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_log_internal_append_truncated_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_log_internal_append_truncated_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_log_internal_append_truncated_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::log_internal_append_truncated PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_log_internal_append_truncated_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_log_internal_append_truncated_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_log_internal_append_truncated_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_log_internal_append_truncated_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_log_internal_append_truncated_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_log_internal_append_truncated_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_log_internal_append_truncated_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_log_internal_append_truncated_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_log_internal_append_truncated_TARGET_PROPERTIES TRUE)

########## COMPONENT log_internal_nullguard TARGET PROPERTIES ######################################

set_property(TARGET absl::log_internal_nullguard PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_log_internal_nullguard_LINK_LIBS_DEBUG}
                ${absl_log_internal_nullguard_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_log_internal_nullguard_LINK_LIBS_RELEASE}
                ${absl_log_internal_nullguard_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_log_internal_nullguard_LINK_LIBS_RELWITHDEBINFO}
                ${absl_log_internal_nullguard_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_log_internal_nullguard_LINK_LIBS_MINSIZEREL}
                ${absl_log_internal_nullguard_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::log_internal_nullguard PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_log_internal_nullguard_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_log_internal_nullguard_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_log_internal_nullguard_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_log_internal_nullguard_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::log_internal_nullguard PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_log_internal_nullguard_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_log_internal_nullguard_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_log_internal_nullguard_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_log_internal_nullguard_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::log_internal_nullguard PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_log_internal_nullguard_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_log_internal_nullguard_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_log_internal_nullguard_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_log_internal_nullguard_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_log_internal_nullguard_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_log_internal_nullguard_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_log_internal_nullguard_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_log_internal_nullguard_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_log_internal_nullguard_TARGET_PROPERTIES TRUE)

########## COMPONENT log_internal_proto TARGET PROPERTIES ######################################

set_property(TARGET absl::log_internal_proto PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_log_internal_proto_LINK_LIBS_DEBUG}
                ${absl_log_internal_proto_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_log_internal_proto_LINK_LIBS_RELEASE}
                ${absl_log_internal_proto_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_log_internal_proto_LINK_LIBS_RELWITHDEBINFO}
                ${absl_log_internal_proto_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_log_internal_proto_LINK_LIBS_MINSIZEREL}
                ${absl_log_internal_proto_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::log_internal_proto PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_log_internal_proto_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_log_internal_proto_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_log_internal_proto_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_log_internal_proto_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::log_internal_proto PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_log_internal_proto_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_log_internal_proto_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_log_internal_proto_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_log_internal_proto_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::log_internal_proto PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_log_internal_proto_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_log_internal_proto_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_log_internal_proto_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_log_internal_proto_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_log_internal_proto_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_log_internal_proto_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_log_internal_proto_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_log_internal_proto_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_log_internal_proto_TARGET_PROPERTIES TRUE)

########## COMPONENT log_internal_format TARGET PROPERTIES ######################################

set_property(TARGET absl::log_internal_format PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_log_internal_format_LINK_LIBS_DEBUG}
                ${absl_log_internal_format_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_log_internal_format_LINK_LIBS_RELEASE}
                ${absl_log_internal_format_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_log_internal_format_LINK_LIBS_RELWITHDEBINFO}
                ${absl_log_internal_format_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_log_internal_format_LINK_LIBS_MINSIZEREL}
                ${absl_log_internal_format_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::log_internal_format PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_log_internal_format_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_log_internal_format_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_log_internal_format_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_log_internal_format_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::log_internal_format PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_log_internal_format_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_log_internal_format_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_log_internal_format_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_log_internal_format_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::log_internal_format PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_log_internal_format_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_log_internal_format_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_log_internal_format_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_log_internal_format_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_log_internal_format_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_log_internal_format_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_log_internal_format_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_log_internal_format_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_log_internal_format_TARGET_PROPERTIES TRUE)

########## COMPONENT examine_stack TARGET PROPERTIES ######################################

set_property(TARGET absl::examine_stack PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_examine_stack_LINK_LIBS_DEBUG}
                ${absl_examine_stack_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_examine_stack_LINK_LIBS_RELEASE}
                ${absl_examine_stack_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_examine_stack_LINK_LIBS_RELWITHDEBINFO}
                ${absl_examine_stack_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_examine_stack_LINK_LIBS_MINSIZEREL}
                ${absl_examine_stack_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::examine_stack PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_examine_stack_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_examine_stack_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_examine_stack_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_examine_stack_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::examine_stack PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_examine_stack_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_examine_stack_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_examine_stack_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_examine_stack_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::examine_stack PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_examine_stack_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_examine_stack_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_examine_stack_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_examine_stack_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_examine_stack_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_examine_stack_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_examine_stack_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_examine_stack_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_examine_stack_TARGET_PROPERTIES TRUE)

########## COMPONENT log_internal_message TARGET PROPERTIES ######################################

set_property(TARGET absl::log_internal_message PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_log_internal_message_LINK_LIBS_DEBUG}
                ${absl_log_internal_message_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_log_internal_message_LINK_LIBS_RELEASE}
                ${absl_log_internal_message_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_log_internal_message_LINK_LIBS_RELWITHDEBINFO}
                ${absl_log_internal_message_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_log_internal_message_LINK_LIBS_MINSIZEREL}
                ${absl_log_internal_message_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::log_internal_message PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_log_internal_message_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_log_internal_message_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_log_internal_message_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_log_internal_message_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::log_internal_message PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_log_internal_message_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_log_internal_message_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_log_internal_message_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_log_internal_message_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::log_internal_message PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_log_internal_message_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_log_internal_message_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_log_internal_message_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_log_internal_message_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_log_internal_message_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_log_internal_message_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_log_internal_message_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_log_internal_message_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_log_internal_message_TARGET_PROPERTIES TRUE)

########## COMPONENT log_internal_structured TARGET PROPERTIES ######################################

set_property(TARGET absl::log_internal_structured PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_log_internal_structured_LINK_LIBS_DEBUG}
                ${absl_log_internal_structured_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_log_internal_structured_LINK_LIBS_RELEASE}
                ${absl_log_internal_structured_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_log_internal_structured_LINK_LIBS_RELWITHDEBINFO}
                ${absl_log_internal_structured_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_log_internal_structured_LINK_LIBS_MINSIZEREL}
                ${absl_log_internal_structured_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::log_internal_structured PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_log_internal_structured_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_log_internal_structured_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_log_internal_structured_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_log_internal_structured_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::log_internal_structured PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_log_internal_structured_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_log_internal_structured_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_log_internal_structured_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_log_internal_structured_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::log_internal_structured PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_log_internal_structured_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_log_internal_structured_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_log_internal_structured_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_log_internal_structured_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_log_internal_structured_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_log_internal_structured_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_log_internal_structured_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_log_internal_structured_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_log_internal_structured_TARGET_PROPERTIES TRUE)

########## COMPONENT log_structured TARGET PROPERTIES ######################################

set_property(TARGET absl::log_structured PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_log_structured_LINK_LIBS_DEBUG}
                ${absl_log_structured_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_log_structured_LINK_LIBS_RELEASE}
                ${absl_log_structured_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_log_structured_LINK_LIBS_RELWITHDEBINFO}
                ${absl_log_structured_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_log_structured_LINK_LIBS_MINSIZEREL}
                ${absl_log_structured_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::log_structured PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_log_structured_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_log_structured_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_log_structured_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_log_structured_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::log_structured PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_log_structured_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_log_structured_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_log_structured_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_log_structured_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::log_structured PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_log_structured_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_log_structured_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_log_structured_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_log_structured_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_log_structured_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_log_structured_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_log_structured_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_log_structured_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_log_structured_TARGET_PROPERTIES TRUE)

########## COMPONENT log_internal_voidify TARGET PROPERTIES ######################################

set_property(TARGET absl::log_internal_voidify PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_log_internal_voidify_LINK_LIBS_DEBUG}
                ${absl_log_internal_voidify_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_log_internal_voidify_LINK_LIBS_RELEASE}
                ${absl_log_internal_voidify_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_log_internal_voidify_LINK_LIBS_RELWITHDEBINFO}
                ${absl_log_internal_voidify_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_log_internal_voidify_LINK_LIBS_MINSIZEREL}
                ${absl_log_internal_voidify_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::log_internal_voidify PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_log_internal_voidify_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_log_internal_voidify_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_log_internal_voidify_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_log_internal_voidify_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::log_internal_voidify PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_log_internal_voidify_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_log_internal_voidify_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_log_internal_voidify_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_log_internal_voidify_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::log_internal_voidify PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_log_internal_voidify_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_log_internal_voidify_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_log_internal_voidify_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_log_internal_voidify_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_log_internal_voidify_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_log_internal_voidify_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_log_internal_voidify_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_log_internal_voidify_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_log_internal_voidify_TARGET_PROPERTIES TRUE)

########## COMPONENT log_internal_nullstream TARGET PROPERTIES ######################################

set_property(TARGET absl::log_internal_nullstream PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_log_internal_nullstream_LINK_LIBS_DEBUG}
                ${absl_log_internal_nullstream_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_log_internal_nullstream_LINK_LIBS_RELEASE}
                ${absl_log_internal_nullstream_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_log_internal_nullstream_LINK_LIBS_RELWITHDEBINFO}
                ${absl_log_internal_nullstream_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_log_internal_nullstream_LINK_LIBS_MINSIZEREL}
                ${absl_log_internal_nullstream_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::log_internal_nullstream PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_log_internal_nullstream_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_log_internal_nullstream_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_log_internal_nullstream_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_log_internal_nullstream_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::log_internal_nullstream PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_log_internal_nullstream_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_log_internal_nullstream_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_log_internal_nullstream_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_log_internal_nullstream_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::log_internal_nullstream PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_log_internal_nullstream_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_log_internal_nullstream_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_log_internal_nullstream_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_log_internal_nullstream_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_log_internal_nullstream_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_log_internal_nullstream_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_log_internal_nullstream_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_log_internal_nullstream_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_log_internal_nullstream_TARGET_PROPERTIES TRUE)

########## COMPONENT log_internal_strip TARGET PROPERTIES ######################################

set_property(TARGET absl::log_internal_strip PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_log_internal_strip_LINK_LIBS_DEBUG}
                ${absl_log_internal_strip_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_log_internal_strip_LINK_LIBS_RELEASE}
                ${absl_log_internal_strip_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_log_internal_strip_LINK_LIBS_RELWITHDEBINFO}
                ${absl_log_internal_strip_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_log_internal_strip_LINK_LIBS_MINSIZEREL}
                ${absl_log_internal_strip_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::log_internal_strip PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_log_internal_strip_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_log_internal_strip_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_log_internal_strip_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_log_internal_strip_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::log_internal_strip PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_log_internal_strip_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_log_internal_strip_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_log_internal_strip_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_log_internal_strip_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::log_internal_strip PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_log_internal_strip_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_log_internal_strip_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_log_internal_strip_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_log_internal_strip_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_log_internal_strip_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_log_internal_strip_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_log_internal_strip_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_log_internal_strip_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_log_internal_strip_TARGET_PROPERTIES TRUE)

########## COMPONENT log_internal_conditions TARGET PROPERTIES ######################################

set_property(TARGET absl::log_internal_conditions PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_log_internal_conditions_LINK_LIBS_DEBUG}
                ${absl_log_internal_conditions_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_log_internal_conditions_LINK_LIBS_RELEASE}
                ${absl_log_internal_conditions_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_log_internal_conditions_LINK_LIBS_RELWITHDEBINFO}
                ${absl_log_internal_conditions_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_log_internal_conditions_LINK_LIBS_MINSIZEREL}
                ${absl_log_internal_conditions_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::log_internal_conditions PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_log_internal_conditions_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_log_internal_conditions_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_log_internal_conditions_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_log_internal_conditions_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::log_internal_conditions PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_log_internal_conditions_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_log_internal_conditions_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_log_internal_conditions_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_log_internal_conditions_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::log_internal_conditions PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_log_internal_conditions_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_log_internal_conditions_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_log_internal_conditions_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_log_internal_conditions_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_log_internal_conditions_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_log_internal_conditions_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_log_internal_conditions_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_log_internal_conditions_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_log_internal_conditions_TARGET_PROPERTIES TRUE)

########## COMPONENT log_internal_log_impl TARGET PROPERTIES ######################################

set_property(TARGET absl::log_internal_log_impl PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_log_internal_log_impl_LINK_LIBS_DEBUG}
                ${absl_log_internal_log_impl_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_log_internal_log_impl_LINK_LIBS_RELEASE}
                ${absl_log_internal_log_impl_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_log_internal_log_impl_LINK_LIBS_RELWITHDEBINFO}
                ${absl_log_internal_log_impl_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_log_internal_log_impl_LINK_LIBS_MINSIZEREL}
                ${absl_log_internal_log_impl_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::log_internal_log_impl PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_log_internal_log_impl_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_log_internal_log_impl_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_log_internal_log_impl_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_log_internal_log_impl_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::log_internal_log_impl PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_log_internal_log_impl_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_log_internal_log_impl_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_log_internal_log_impl_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_log_internal_log_impl_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::log_internal_log_impl PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_log_internal_log_impl_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_log_internal_log_impl_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_log_internal_log_impl_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_log_internal_log_impl_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_log_internal_log_impl_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_log_internal_log_impl_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_log_internal_log_impl_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_log_internal_log_impl_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_log_internal_log_impl_TARGET_PROPERTIES TRUE)

########## COMPONENT absl_log TARGET PROPERTIES ######################################

set_property(TARGET absl::absl_log PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_absl_log_LINK_LIBS_DEBUG}
                ${absl_absl_log_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_absl_log_LINK_LIBS_RELEASE}
                ${absl_absl_log_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_absl_log_LINK_LIBS_RELWITHDEBINFO}
                ${absl_absl_log_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_absl_log_LINK_LIBS_MINSIZEREL}
                ${absl_absl_log_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::absl_log PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_absl_log_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_absl_log_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_absl_log_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_absl_log_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::absl_log PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_absl_log_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_absl_log_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_absl_log_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_absl_log_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::absl_log PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_absl_log_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_absl_log_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_absl_log_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_absl_log_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_absl_log_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_absl_log_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_absl_log_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_absl_log_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_absl_log_TARGET_PROPERTIES TRUE)

########## COMPONENT log_streamer TARGET PROPERTIES ######################################

set_property(TARGET absl::log_streamer PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_log_streamer_LINK_LIBS_DEBUG}
                ${absl_log_streamer_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_log_streamer_LINK_LIBS_RELEASE}
                ${absl_log_streamer_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_log_streamer_LINK_LIBS_RELWITHDEBINFO}
                ${absl_log_streamer_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_log_streamer_LINK_LIBS_MINSIZEREL}
                ${absl_log_streamer_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::log_streamer PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_log_streamer_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_log_streamer_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_log_streamer_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_log_streamer_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::log_streamer PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_log_streamer_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_log_streamer_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_log_streamer_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_log_streamer_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::log_streamer PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_log_streamer_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_log_streamer_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_log_streamer_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_log_streamer_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_log_streamer_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_log_streamer_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_log_streamer_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_log_streamer_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_log_streamer_TARGET_PROPERTIES TRUE)

########## COMPONENT log TARGET PROPERTIES ######################################

set_property(TARGET absl::log PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_log_LINK_LIBS_DEBUG}
                ${absl_log_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_log_LINK_LIBS_RELEASE}
                ${absl_log_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_log_LINK_LIBS_RELWITHDEBINFO}
                ${absl_log_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_log_LINK_LIBS_MINSIZEREL}
                ${absl_log_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::log PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_log_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_log_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_log_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_log_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::log PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_log_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_log_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_log_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_log_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::log PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_log_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_log_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_log_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_log_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_log_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_log_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_log_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_log_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_log_TARGET_PROPERTIES TRUE)

########## COMPONENT log_initialize TARGET PROPERTIES ######################################

set_property(TARGET absl::log_initialize PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_log_initialize_LINK_LIBS_DEBUG}
                ${absl_log_initialize_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_log_initialize_LINK_LIBS_RELEASE}
                ${absl_log_initialize_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_log_initialize_LINK_LIBS_RELWITHDEBINFO}
                ${absl_log_initialize_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_log_initialize_LINK_LIBS_MINSIZEREL}
                ${absl_log_initialize_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::log_initialize PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_log_initialize_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_log_initialize_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_log_initialize_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_log_initialize_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::log_initialize PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_log_initialize_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_log_initialize_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_log_initialize_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_log_initialize_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::log_initialize PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_log_initialize_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_log_initialize_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_log_initialize_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_log_initialize_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_log_initialize_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_log_initialize_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_log_initialize_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_log_initialize_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_log_initialize_TARGET_PROPERTIES TRUE)

########## COMPONENT flags_commandlineflag_internal TARGET PROPERTIES ######################################

set_property(TARGET absl::flags_commandlineflag_internal PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_flags_commandlineflag_internal_LINK_LIBS_DEBUG}
                ${absl_flags_commandlineflag_internal_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_flags_commandlineflag_internal_LINK_LIBS_RELEASE}
                ${absl_flags_commandlineflag_internal_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_flags_commandlineflag_internal_LINK_LIBS_RELWITHDEBINFO}
                ${absl_flags_commandlineflag_internal_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_flags_commandlineflag_internal_LINK_LIBS_MINSIZEREL}
                ${absl_flags_commandlineflag_internal_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::flags_commandlineflag_internal PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_flags_commandlineflag_internal_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_flags_commandlineflag_internal_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_flags_commandlineflag_internal_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_flags_commandlineflag_internal_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::flags_commandlineflag_internal PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_flags_commandlineflag_internal_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_flags_commandlineflag_internal_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_flags_commandlineflag_internal_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_flags_commandlineflag_internal_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::flags_commandlineflag_internal PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_flags_commandlineflag_internal_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_flags_commandlineflag_internal_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_flags_commandlineflag_internal_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_flags_commandlineflag_internal_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_flags_commandlineflag_internal_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_flags_commandlineflag_internal_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_flags_commandlineflag_internal_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_flags_commandlineflag_internal_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_flags_commandlineflag_internal_TARGET_PROPERTIES TRUE)

########## COMPONENT flags_commandlineflag TARGET PROPERTIES ######################################

set_property(TARGET absl::flags_commandlineflag PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_flags_commandlineflag_LINK_LIBS_DEBUG}
                ${absl_flags_commandlineflag_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_flags_commandlineflag_LINK_LIBS_RELEASE}
                ${absl_flags_commandlineflag_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_flags_commandlineflag_LINK_LIBS_RELWITHDEBINFO}
                ${absl_flags_commandlineflag_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_flags_commandlineflag_LINK_LIBS_MINSIZEREL}
                ${absl_flags_commandlineflag_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::flags_commandlineflag PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_flags_commandlineflag_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_flags_commandlineflag_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_flags_commandlineflag_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_flags_commandlineflag_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::flags_commandlineflag PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_flags_commandlineflag_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_flags_commandlineflag_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_flags_commandlineflag_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_flags_commandlineflag_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::flags_commandlineflag PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_flags_commandlineflag_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_flags_commandlineflag_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_flags_commandlineflag_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_flags_commandlineflag_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_flags_commandlineflag_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_flags_commandlineflag_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_flags_commandlineflag_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_flags_commandlineflag_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_flags_commandlineflag_TARGET_PROPERTIES TRUE)

########## COMPONENT flags_marshalling TARGET PROPERTIES ######################################

set_property(TARGET absl::flags_marshalling PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_flags_marshalling_LINK_LIBS_DEBUG}
                ${absl_flags_marshalling_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_flags_marshalling_LINK_LIBS_RELEASE}
                ${absl_flags_marshalling_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_flags_marshalling_LINK_LIBS_RELWITHDEBINFO}
                ${absl_flags_marshalling_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_flags_marshalling_LINK_LIBS_MINSIZEREL}
                ${absl_flags_marshalling_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::flags_marshalling PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_flags_marshalling_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_flags_marshalling_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_flags_marshalling_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_flags_marshalling_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::flags_marshalling PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_flags_marshalling_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_flags_marshalling_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_flags_marshalling_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_flags_marshalling_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::flags_marshalling PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_flags_marshalling_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_flags_marshalling_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_flags_marshalling_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_flags_marshalling_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_flags_marshalling_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_flags_marshalling_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_flags_marshalling_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_flags_marshalling_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_flags_marshalling_TARGET_PROPERTIES TRUE)

########## COMPONENT flags_path_util TARGET PROPERTIES ######################################

set_property(TARGET absl::flags_path_util PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_flags_path_util_LINK_LIBS_DEBUG}
                ${absl_flags_path_util_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_flags_path_util_LINK_LIBS_RELEASE}
                ${absl_flags_path_util_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_flags_path_util_LINK_LIBS_RELWITHDEBINFO}
                ${absl_flags_path_util_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_flags_path_util_LINK_LIBS_MINSIZEREL}
                ${absl_flags_path_util_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::flags_path_util PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_flags_path_util_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_flags_path_util_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_flags_path_util_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_flags_path_util_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::flags_path_util PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_flags_path_util_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_flags_path_util_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_flags_path_util_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_flags_path_util_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::flags_path_util PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_flags_path_util_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_flags_path_util_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_flags_path_util_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_flags_path_util_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_flags_path_util_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_flags_path_util_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_flags_path_util_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_flags_path_util_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_flags_path_util_TARGET_PROPERTIES TRUE)

########## COMPONENT flags_program_name TARGET PROPERTIES ######################################

set_property(TARGET absl::flags_program_name PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_flags_program_name_LINK_LIBS_DEBUG}
                ${absl_flags_program_name_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_flags_program_name_LINK_LIBS_RELEASE}
                ${absl_flags_program_name_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_flags_program_name_LINK_LIBS_RELWITHDEBINFO}
                ${absl_flags_program_name_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_flags_program_name_LINK_LIBS_MINSIZEREL}
                ${absl_flags_program_name_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::flags_program_name PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_flags_program_name_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_flags_program_name_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_flags_program_name_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_flags_program_name_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::flags_program_name PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_flags_program_name_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_flags_program_name_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_flags_program_name_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_flags_program_name_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::flags_program_name PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_flags_program_name_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_flags_program_name_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_flags_program_name_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_flags_program_name_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_flags_program_name_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_flags_program_name_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_flags_program_name_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_flags_program_name_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_flags_program_name_TARGET_PROPERTIES TRUE)

########## COMPONENT flags_config TARGET PROPERTIES ######################################

set_property(TARGET absl::flags_config PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_flags_config_LINK_LIBS_DEBUG}
                ${absl_flags_config_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_flags_config_LINK_LIBS_RELEASE}
                ${absl_flags_config_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_flags_config_LINK_LIBS_RELWITHDEBINFO}
                ${absl_flags_config_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_flags_config_LINK_LIBS_MINSIZEREL}
                ${absl_flags_config_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::flags_config PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_flags_config_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_flags_config_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_flags_config_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_flags_config_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::flags_config PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_flags_config_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_flags_config_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_flags_config_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_flags_config_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::flags_config PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_flags_config_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_flags_config_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_flags_config_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_flags_config_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_flags_config_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_flags_config_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_flags_config_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_flags_config_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_flags_config_TARGET_PROPERTIES TRUE)

########## COMPONENT flags_internal TARGET PROPERTIES ######################################

set_property(TARGET absl::flags_internal PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_flags_internal_LINK_LIBS_DEBUG}
                ${absl_flags_internal_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_flags_internal_LINK_LIBS_RELEASE}
                ${absl_flags_internal_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_flags_internal_LINK_LIBS_RELWITHDEBINFO}
                ${absl_flags_internal_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_flags_internal_LINK_LIBS_MINSIZEREL}
                ${absl_flags_internal_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::flags_internal PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_flags_internal_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_flags_internal_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_flags_internal_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_flags_internal_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::flags_internal PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_flags_internal_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_flags_internal_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_flags_internal_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_flags_internal_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::flags_internal PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_flags_internal_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_flags_internal_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_flags_internal_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_flags_internal_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_flags_internal_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_flags_internal_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_flags_internal_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_flags_internal_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_flags_internal_TARGET_PROPERTIES TRUE)

########## COMPONENT flags_private_handle_accessor TARGET PROPERTIES ######################################

set_property(TARGET absl::flags_private_handle_accessor PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_flags_private_handle_accessor_LINK_LIBS_DEBUG}
                ${absl_flags_private_handle_accessor_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_flags_private_handle_accessor_LINK_LIBS_RELEASE}
                ${absl_flags_private_handle_accessor_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_flags_private_handle_accessor_LINK_LIBS_RELWITHDEBINFO}
                ${absl_flags_private_handle_accessor_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_flags_private_handle_accessor_LINK_LIBS_MINSIZEREL}
                ${absl_flags_private_handle_accessor_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::flags_private_handle_accessor PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_flags_private_handle_accessor_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_flags_private_handle_accessor_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_flags_private_handle_accessor_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_flags_private_handle_accessor_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::flags_private_handle_accessor PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_flags_private_handle_accessor_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_flags_private_handle_accessor_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_flags_private_handle_accessor_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_flags_private_handle_accessor_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::flags_private_handle_accessor PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_flags_private_handle_accessor_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_flags_private_handle_accessor_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_flags_private_handle_accessor_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_flags_private_handle_accessor_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_flags_private_handle_accessor_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_flags_private_handle_accessor_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_flags_private_handle_accessor_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_flags_private_handle_accessor_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_flags_private_handle_accessor_TARGET_PROPERTIES TRUE)

########## COMPONENT container_common TARGET PROPERTIES ######################################

set_property(TARGET absl::container_common PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_container_common_LINK_LIBS_DEBUG}
                ${absl_container_common_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_container_common_LINK_LIBS_RELEASE}
                ${absl_container_common_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_container_common_LINK_LIBS_RELWITHDEBINFO}
                ${absl_container_common_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_container_common_LINK_LIBS_MINSIZEREL}
                ${absl_container_common_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::container_common PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_container_common_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_container_common_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_container_common_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_container_common_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::container_common PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_container_common_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_container_common_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_container_common_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_container_common_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::container_common PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_container_common_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_container_common_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_container_common_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_container_common_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_container_common_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_container_common_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_container_common_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_container_common_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_container_common_TARGET_PROPERTIES TRUE)

########## COMPONENT hashtable_debug_hooks TARGET PROPERTIES ######################################

set_property(TARGET absl::hashtable_debug_hooks PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_hashtable_debug_hooks_LINK_LIBS_DEBUG}
                ${absl_hashtable_debug_hooks_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_hashtable_debug_hooks_LINK_LIBS_RELEASE}
                ${absl_hashtable_debug_hooks_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_hashtable_debug_hooks_LINK_LIBS_RELWITHDEBINFO}
                ${absl_hashtable_debug_hooks_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_hashtable_debug_hooks_LINK_LIBS_MINSIZEREL}
                ${absl_hashtable_debug_hooks_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::hashtable_debug_hooks PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_hashtable_debug_hooks_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_hashtable_debug_hooks_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_hashtable_debug_hooks_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_hashtable_debug_hooks_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::hashtable_debug_hooks PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_hashtable_debug_hooks_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_hashtable_debug_hooks_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_hashtable_debug_hooks_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_hashtable_debug_hooks_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::hashtable_debug_hooks PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_hashtable_debug_hooks_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_hashtable_debug_hooks_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_hashtable_debug_hooks_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_hashtable_debug_hooks_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_hashtable_debug_hooks_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_hashtable_debug_hooks_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_hashtable_debug_hooks_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_hashtable_debug_hooks_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_hashtable_debug_hooks_TARGET_PROPERTIES TRUE)

########## COMPONENT hashtablez_sampler TARGET PROPERTIES ######################################

set_property(TARGET absl::hashtablez_sampler PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_hashtablez_sampler_LINK_LIBS_DEBUG}
                ${absl_hashtablez_sampler_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_hashtablez_sampler_LINK_LIBS_RELEASE}
                ${absl_hashtablez_sampler_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_hashtablez_sampler_LINK_LIBS_RELWITHDEBINFO}
                ${absl_hashtablez_sampler_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_hashtablez_sampler_LINK_LIBS_MINSIZEREL}
                ${absl_hashtablez_sampler_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::hashtablez_sampler PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_hashtablez_sampler_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_hashtablez_sampler_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_hashtablez_sampler_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_hashtablez_sampler_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::hashtablez_sampler PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_hashtablez_sampler_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_hashtablez_sampler_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_hashtablez_sampler_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_hashtablez_sampler_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::hashtablez_sampler PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_hashtablez_sampler_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_hashtablez_sampler_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_hashtablez_sampler_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_hashtablez_sampler_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_hashtablez_sampler_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_hashtablez_sampler_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_hashtablez_sampler_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_hashtablez_sampler_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_hashtablez_sampler_TARGET_PROPERTIES TRUE)

########## COMPONENT common_policy_traits TARGET PROPERTIES ######################################

set_property(TARGET absl::common_policy_traits PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_common_policy_traits_LINK_LIBS_DEBUG}
                ${absl_common_policy_traits_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_common_policy_traits_LINK_LIBS_RELEASE}
                ${absl_common_policy_traits_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_common_policy_traits_LINK_LIBS_RELWITHDEBINFO}
                ${absl_common_policy_traits_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_common_policy_traits_LINK_LIBS_MINSIZEREL}
                ${absl_common_policy_traits_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::common_policy_traits PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_common_policy_traits_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_common_policy_traits_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_common_policy_traits_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_common_policy_traits_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::common_policy_traits PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_common_policy_traits_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_common_policy_traits_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_common_policy_traits_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_common_policy_traits_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::common_policy_traits PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_common_policy_traits_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_common_policy_traits_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_common_policy_traits_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_common_policy_traits_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_common_policy_traits_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_common_policy_traits_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_common_policy_traits_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_common_policy_traits_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_common_policy_traits_TARGET_PROPERTIES TRUE)

########## COMPONENT hash_policy_traits TARGET PROPERTIES ######################################

set_property(TARGET absl::hash_policy_traits PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_hash_policy_traits_LINK_LIBS_DEBUG}
                ${absl_hash_policy_traits_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_hash_policy_traits_LINK_LIBS_RELEASE}
                ${absl_hash_policy_traits_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_hash_policy_traits_LINK_LIBS_RELWITHDEBINFO}
                ${absl_hash_policy_traits_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_hash_policy_traits_LINK_LIBS_MINSIZEREL}
                ${absl_hash_policy_traits_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::hash_policy_traits PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_hash_policy_traits_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_hash_policy_traits_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_hash_policy_traits_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_hash_policy_traits_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::hash_policy_traits PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_hash_policy_traits_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_hash_policy_traits_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_hash_policy_traits_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_hash_policy_traits_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::hash_policy_traits PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_hash_policy_traits_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_hash_policy_traits_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_hash_policy_traits_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_hash_policy_traits_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_hash_policy_traits_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_hash_policy_traits_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_hash_policy_traits_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_hash_policy_traits_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_hash_policy_traits_TARGET_PROPERTIES TRUE)

########## COMPONENT raw_hash_set TARGET PROPERTIES ######################################

set_property(TARGET absl::raw_hash_set PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_raw_hash_set_LINK_LIBS_DEBUG}
                ${absl_raw_hash_set_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_raw_hash_set_LINK_LIBS_RELEASE}
                ${absl_raw_hash_set_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_raw_hash_set_LINK_LIBS_RELWITHDEBINFO}
                ${absl_raw_hash_set_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_raw_hash_set_LINK_LIBS_MINSIZEREL}
                ${absl_raw_hash_set_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::raw_hash_set PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_raw_hash_set_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_raw_hash_set_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_raw_hash_set_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_raw_hash_set_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::raw_hash_set PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_raw_hash_set_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_raw_hash_set_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_raw_hash_set_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_raw_hash_set_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::raw_hash_set PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_raw_hash_set_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_raw_hash_set_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_raw_hash_set_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_raw_hash_set_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_raw_hash_set_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_raw_hash_set_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_raw_hash_set_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_raw_hash_set_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_raw_hash_set_TARGET_PROPERTIES TRUE)

########## COMPONENT raw_hash_map TARGET PROPERTIES ######################################

set_property(TARGET absl::raw_hash_map PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_raw_hash_map_LINK_LIBS_DEBUG}
                ${absl_raw_hash_map_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_raw_hash_map_LINK_LIBS_RELEASE}
                ${absl_raw_hash_map_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_raw_hash_map_LINK_LIBS_RELWITHDEBINFO}
                ${absl_raw_hash_map_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_raw_hash_map_LINK_LIBS_MINSIZEREL}
                ${absl_raw_hash_map_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::raw_hash_map PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_raw_hash_map_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_raw_hash_map_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_raw_hash_map_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_raw_hash_map_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::raw_hash_map PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_raw_hash_map_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_raw_hash_map_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_raw_hash_map_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_raw_hash_map_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::raw_hash_map PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_raw_hash_map_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_raw_hash_map_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_raw_hash_map_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_raw_hash_map_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_raw_hash_map_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_raw_hash_map_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_raw_hash_map_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_raw_hash_map_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_raw_hash_map_TARGET_PROPERTIES TRUE)

########## COMPONENT hash_function_defaults TARGET PROPERTIES ######################################

set_property(TARGET absl::hash_function_defaults PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_hash_function_defaults_LINK_LIBS_DEBUG}
                ${absl_hash_function_defaults_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_hash_function_defaults_LINK_LIBS_RELEASE}
                ${absl_hash_function_defaults_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_hash_function_defaults_LINK_LIBS_RELWITHDEBINFO}
                ${absl_hash_function_defaults_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_hash_function_defaults_LINK_LIBS_MINSIZEREL}
                ${absl_hash_function_defaults_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::hash_function_defaults PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_hash_function_defaults_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_hash_function_defaults_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_hash_function_defaults_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_hash_function_defaults_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::hash_function_defaults PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_hash_function_defaults_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_hash_function_defaults_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_hash_function_defaults_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_hash_function_defaults_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::hash_function_defaults PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_hash_function_defaults_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_hash_function_defaults_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_hash_function_defaults_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_hash_function_defaults_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_hash_function_defaults_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_hash_function_defaults_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_hash_function_defaults_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_hash_function_defaults_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_hash_function_defaults_TARGET_PROPERTIES TRUE)

########## COMPONENT algorithm_container TARGET PROPERTIES ######################################

set_property(TARGET absl::algorithm_container PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_algorithm_container_LINK_LIBS_DEBUG}
                ${absl_algorithm_container_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_algorithm_container_LINK_LIBS_RELEASE}
                ${absl_algorithm_container_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_algorithm_container_LINK_LIBS_RELWITHDEBINFO}
                ${absl_algorithm_container_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_algorithm_container_LINK_LIBS_MINSIZEREL}
                ${absl_algorithm_container_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::algorithm_container PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_algorithm_container_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_algorithm_container_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_algorithm_container_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_algorithm_container_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::algorithm_container PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_algorithm_container_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_algorithm_container_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_algorithm_container_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_algorithm_container_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::algorithm_container PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_algorithm_container_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_algorithm_container_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_algorithm_container_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_algorithm_container_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_algorithm_container_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_algorithm_container_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_algorithm_container_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_algorithm_container_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_algorithm_container_TARGET_PROPERTIES TRUE)

########## COMPONENT flat_hash_map TARGET PROPERTIES ######################################

set_property(TARGET absl::flat_hash_map PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_flat_hash_map_LINK_LIBS_DEBUG}
                ${absl_flat_hash_map_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_flat_hash_map_LINK_LIBS_RELEASE}
                ${absl_flat_hash_map_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_flat_hash_map_LINK_LIBS_RELWITHDEBINFO}
                ${absl_flat_hash_map_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_flat_hash_map_LINK_LIBS_MINSIZEREL}
                ${absl_flat_hash_map_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::flat_hash_map PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_flat_hash_map_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_flat_hash_map_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_flat_hash_map_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_flat_hash_map_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::flat_hash_map PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_flat_hash_map_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_flat_hash_map_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_flat_hash_map_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_flat_hash_map_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::flat_hash_map PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_flat_hash_map_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_flat_hash_map_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_flat_hash_map_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_flat_hash_map_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_flat_hash_map_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_flat_hash_map_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_flat_hash_map_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_flat_hash_map_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_flat_hash_map_TARGET_PROPERTIES TRUE)

########## COMPONENT flags_reflection TARGET PROPERTIES ######################################

set_property(TARGET absl::flags_reflection PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_flags_reflection_LINK_LIBS_DEBUG}
                ${absl_flags_reflection_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_flags_reflection_LINK_LIBS_RELEASE}
                ${absl_flags_reflection_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_flags_reflection_LINK_LIBS_RELWITHDEBINFO}
                ${absl_flags_reflection_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_flags_reflection_LINK_LIBS_MINSIZEREL}
                ${absl_flags_reflection_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::flags_reflection PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_flags_reflection_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_flags_reflection_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_flags_reflection_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_flags_reflection_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::flags_reflection PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_flags_reflection_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_flags_reflection_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_flags_reflection_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_flags_reflection_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::flags_reflection PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_flags_reflection_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_flags_reflection_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_flags_reflection_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_flags_reflection_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_flags_reflection_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_flags_reflection_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_flags_reflection_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_flags_reflection_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_flags_reflection_TARGET_PROPERTIES TRUE)

########## COMPONENT flags TARGET PROPERTIES ######################################

set_property(TARGET absl::flags PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_flags_LINK_LIBS_DEBUG}
                ${absl_flags_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_flags_LINK_LIBS_RELEASE}
                ${absl_flags_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_flags_LINK_LIBS_RELWITHDEBINFO}
                ${absl_flags_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_flags_LINK_LIBS_MINSIZEREL}
                ${absl_flags_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::flags PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_flags_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_flags_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_flags_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_flags_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::flags PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_flags_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_flags_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_flags_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_flags_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::flags PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_flags_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_flags_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_flags_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_flags_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_flags_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_flags_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_flags_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_flags_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_flags_TARGET_PROPERTIES TRUE)

########## COMPONENT log_internal_flags TARGET PROPERTIES ######################################

set_property(TARGET absl::log_internal_flags PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_log_internal_flags_LINK_LIBS_DEBUG}
                ${absl_log_internal_flags_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_log_internal_flags_LINK_LIBS_RELEASE}
                ${absl_log_internal_flags_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_log_internal_flags_LINK_LIBS_RELWITHDEBINFO}
                ${absl_log_internal_flags_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_log_internal_flags_LINK_LIBS_MINSIZEREL}
                ${absl_log_internal_flags_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::log_internal_flags PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_log_internal_flags_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_log_internal_flags_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_log_internal_flags_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_log_internal_flags_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::log_internal_flags PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_log_internal_flags_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_log_internal_flags_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_log_internal_flags_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_log_internal_flags_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::log_internal_flags PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_log_internal_flags_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_log_internal_flags_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_log_internal_flags_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_log_internal_flags_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_log_internal_flags_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_log_internal_flags_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_log_internal_flags_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_log_internal_flags_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_log_internal_flags_TARGET_PROPERTIES TRUE)

########## COMPONENT log_flags TARGET PROPERTIES ######################################

set_property(TARGET absl::log_flags PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_log_flags_LINK_LIBS_DEBUG}
                ${absl_log_flags_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_log_flags_LINK_LIBS_RELEASE}
                ${absl_log_flags_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_log_flags_LINK_LIBS_RELWITHDEBINFO}
                ${absl_log_flags_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_log_flags_LINK_LIBS_MINSIZEREL}
                ${absl_log_flags_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::log_flags PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_log_flags_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_log_flags_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_log_flags_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_log_flags_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::log_flags PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_log_flags_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_log_flags_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_log_flags_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_log_flags_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::log_flags PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_log_flags_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_log_flags_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_log_flags_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_log_flags_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_log_flags_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_log_flags_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_log_flags_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_log_flags_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_log_flags_TARGET_PROPERTIES TRUE)

########## COMPONENT die_if_null TARGET PROPERTIES ######################################

set_property(TARGET absl::die_if_null PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_die_if_null_LINK_LIBS_DEBUG}
                ${absl_die_if_null_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_die_if_null_LINK_LIBS_RELEASE}
                ${absl_die_if_null_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_die_if_null_LINK_LIBS_RELWITHDEBINFO}
                ${absl_die_if_null_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_die_if_null_LINK_LIBS_MINSIZEREL}
                ${absl_die_if_null_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::die_if_null PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_die_if_null_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_die_if_null_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_die_if_null_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_die_if_null_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::die_if_null PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_die_if_null_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_die_if_null_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_die_if_null_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_die_if_null_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::die_if_null PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_die_if_null_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_die_if_null_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_die_if_null_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_die_if_null_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_die_if_null_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_die_if_null_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_die_if_null_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_die_if_null_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_die_if_null_TARGET_PROPERTIES TRUE)

########## COMPONENT log_internal_check_op TARGET PROPERTIES ######################################

set_property(TARGET absl::log_internal_check_op PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_log_internal_check_op_LINK_LIBS_DEBUG}
                ${absl_log_internal_check_op_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_log_internal_check_op_LINK_LIBS_RELEASE}
                ${absl_log_internal_check_op_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_log_internal_check_op_LINK_LIBS_RELWITHDEBINFO}
                ${absl_log_internal_check_op_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_log_internal_check_op_LINK_LIBS_MINSIZEREL}
                ${absl_log_internal_check_op_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::log_internal_check_op PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_log_internal_check_op_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_log_internal_check_op_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_log_internal_check_op_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_log_internal_check_op_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::log_internal_check_op PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_log_internal_check_op_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_log_internal_check_op_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_log_internal_check_op_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_log_internal_check_op_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::log_internal_check_op PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_log_internal_check_op_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_log_internal_check_op_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_log_internal_check_op_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_log_internal_check_op_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_log_internal_check_op_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_log_internal_check_op_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_log_internal_check_op_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_log_internal_check_op_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_log_internal_check_op_TARGET_PROPERTIES TRUE)

########## COMPONENT log_internal_check_impl TARGET PROPERTIES ######################################

set_property(TARGET absl::log_internal_check_impl PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_log_internal_check_impl_LINK_LIBS_DEBUG}
                ${absl_log_internal_check_impl_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_log_internal_check_impl_LINK_LIBS_RELEASE}
                ${absl_log_internal_check_impl_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_log_internal_check_impl_LINK_LIBS_RELWITHDEBINFO}
                ${absl_log_internal_check_impl_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_log_internal_check_impl_LINK_LIBS_MINSIZEREL}
                ${absl_log_internal_check_impl_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::log_internal_check_impl PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_log_internal_check_impl_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_log_internal_check_impl_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_log_internal_check_impl_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_log_internal_check_impl_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::log_internal_check_impl PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_log_internal_check_impl_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_log_internal_check_impl_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_log_internal_check_impl_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_log_internal_check_impl_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::log_internal_check_impl PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_log_internal_check_impl_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_log_internal_check_impl_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_log_internal_check_impl_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_log_internal_check_impl_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_log_internal_check_impl_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_log_internal_check_impl_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_log_internal_check_impl_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_log_internal_check_impl_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_log_internal_check_impl_TARGET_PROPERTIES TRUE)

########## COMPONENT check TARGET PROPERTIES ######################################

set_property(TARGET absl::check PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_check_LINK_LIBS_DEBUG}
                ${absl_check_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_check_LINK_LIBS_RELEASE}
                ${absl_check_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_check_LINK_LIBS_RELWITHDEBINFO}
                ${absl_check_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_check_LINK_LIBS_MINSIZEREL}
                ${absl_check_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::check PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_check_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_check_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_check_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_check_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::check PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_check_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_check_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_check_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_check_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::check PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_check_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_check_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_check_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_check_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_check_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_check_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_check_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_check_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_check_TARGET_PROPERTIES TRUE)

########## COMPONENT absl_check TARGET PROPERTIES ######################################

set_property(TARGET absl::absl_check PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_absl_check_LINK_LIBS_DEBUG}
                ${absl_absl_check_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_absl_check_LINK_LIBS_RELEASE}
                ${absl_absl_check_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_absl_check_LINK_LIBS_RELWITHDEBINFO}
                ${absl_absl_check_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_absl_check_LINK_LIBS_MINSIZEREL}
                ${absl_absl_check_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::absl_check PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_absl_check_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_absl_check_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_absl_check_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_absl_check_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::absl_check PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_absl_check_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_absl_check_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_absl_check_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_absl_check_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::absl_check PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_absl_check_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_absl_check_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_absl_check_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_absl_check_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_absl_check_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_absl_check_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_absl_check_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_absl_check_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_absl_check_TARGET_PROPERTIES TRUE)

########## COMPONENT bind_front TARGET PROPERTIES ######################################

set_property(TARGET absl::bind_front PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_bind_front_LINK_LIBS_DEBUG}
                ${absl_bind_front_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_bind_front_LINK_LIBS_RELEASE}
                ${absl_bind_front_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_bind_front_LINK_LIBS_RELWITHDEBINFO}
                ${absl_bind_front_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_bind_front_LINK_LIBS_MINSIZEREL}
                ${absl_bind_front_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::bind_front PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_bind_front_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_bind_front_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_bind_front_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_bind_front_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::bind_front PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_bind_front_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_bind_front_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_bind_front_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_bind_front_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::bind_front PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_bind_front_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_bind_front_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_bind_front_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_bind_front_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_bind_front_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_bind_front_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_bind_front_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_bind_front_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_bind_front_TARGET_PROPERTIES TRUE)

########## COMPONENT any_invocable TARGET PROPERTIES ######################################

set_property(TARGET absl::any_invocable PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_any_invocable_LINK_LIBS_DEBUG}
                ${absl_any_invocable_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_any_invocable_LINK_LIBS_RELEASE}
                ${absl_any_invocable_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_any_invocable_LINK_LIBS_RELWITHDEBINFO}
                ${absl_any_invocable_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_any_invocable_LINK_LIBS_MINSIZEREL}
                ${absl_any_invocable_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::any_invocable PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_any_invocable_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_any_invocable_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_any_invocable_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_any_invocable_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::any_invocable PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_any_invocable_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_any_invocable_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_any_invocable_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_any_invocable_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::any_invocable PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_any_invocable_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_any_invocable_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_any_invocable_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_any_invocable_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_any_invocable_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_any_invocable_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_any_invocable_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_any_invocable_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_any_invocable_TARGET_PROPERTIES TRUE)

########## COMPONENT flags_usage_internal TARGET PROPERTIES ######################################

set_property(TARGET absl::flags_usage_internal PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_flags_usage_internal_LINK_LIBS_DEBUG}
                ${absl_flags_usage_internal_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_flags_usage_internal_LINK_LIBS_RELEASE}
                ${absl_flags_usage_internal_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_flags_usage_internal_LINK_LIBS_RELWITHDEBINFO}
                ${absl_flags_usage_internal_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_flags_usage_internal_LINK_LIBS_MINSIZEREL}
                ${absl_flags_usage_internal_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::flags_usage_internal PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_flags_usage_internal_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_flags_usage_internal_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_flags_usage_internal_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_flags_usage_internal_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::flags_usage_internal PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_flags_usage_internal_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_flags_usage_internal_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_flags_usage_internal_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_flags_usage_internal_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::flags_usage_internal PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_flags_usage_internal_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_flags_usage_internal_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_flags_usage_internal_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_flags_usage_internal_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_flags_usage_internal_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_flags_usage_internal_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_flags_usage_internal_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_flags_usage_internal_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_flags_usage_internal_TARGET_PROPERTIES TRUE)

########## COMPONENT flags_usage TARGET PROPERTIES ######################################

set_property(TARGET absl::flags_usage PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_flags_usage_LINK_LIBS_DEBUG}
                ${absl_flags_usage_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_flags_usage_LINK_LIBS_RELEASE}
                ${absl_flags_usage_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_flags_usage_LINK_LIBS_RELWITHDEBINFO}
                ${absl_flags_usage_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_flags_usage_LINK_LIBS_MINSIZEREL}
                ${absl_flags_usage_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::flags_usage PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_flags_usage_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_flags_usage_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_flags_usage_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_flags_usage_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::flags_usage PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_flags_usage_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_flags_usage_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_flags_usage_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_flags_usage_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::flags_usage PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_flags_usage_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_flags_usage_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_flags_usage_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_flags_usage_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_flags_usage_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_flags_usage_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_flags_usage_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_flags_usage_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_flags_usage_TARGET_PROPERTIES TRUE)

########## COMPONENT flags_parse TARGET PROPERTIES ######################################

set_property(TARGET absl::flags_parse PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_flags_parse_LINK_LIBS_DEBUG}
                ${absl_flags_parse_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_flags_parse_LINK_LIBS_RELEASE}
                ${absl_flags_parse_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_flags_parse_LINK_LIBS_RELWITHDEBINFO}
                ${absl_flags_parse_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_flags_parse_LINK_LIBS_MINSIZEREL}
                ${absl_flags_parse_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::flags_parse PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_flags_parse_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_flags_parse_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_flags_parse_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_flags_parse_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::flags_parse PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_flags_parse_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_flags_parse_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_flags_parse_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_flags_parse_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::flags_parse PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_flags_parse_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_flags_parse_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_flags_parse_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_flags_parse_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_flags_parse_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_flags_parse_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_flags_parse_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_flags_parse_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_flags_parse_TARGET_PROPERTIES TRUE)

########## COMPONENT leak_check TARGET PROPERTIES ######################################

set_property(TARGET absl::leak_check PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_leak_check_LINK_LIBS_DEBUG}
                ${absl_leak_check_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_leak_check_LINK_LIBS_RELEASE}
                ${absl_leak_check_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_leak_check_LINK_LIBS_RELWITHDEBINFO}
                ${absl_leak_check_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_leak_check_LINK_LIBS_MINSIZEREL}
                ${absl_leak_check_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::leak_check PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_leak_check_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_leak_check_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_leak_check_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_leak_check_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::leak_check PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_leak_check_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_leak_check_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_leak_check_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_leak_check_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::leak_check PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_leak_check_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_leak_check_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_leak_check_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_leak_check_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_leak_check_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_leak_check_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_leak_check_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_leak_check_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_leak_check_TARGET_PROPERTIES TRUE)

########## COMPONENT debugging TARGET PROPERTIES ######################################

set_property(TARGET absl::debugging PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_debugging_LINK_LIBS_DEBUG}
                ${absl_debugging_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_debugging_LINK_LIBS_RELEASE}
                ${absl_debugging_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_debugging_LINK_LIBS_RELWITHDEBINFO}
                ${absl_debugging_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_debugging_LINK_LIBS_MINSIZEREL}
                ${absl_debugging_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::debugging PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_debugging_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_debugging_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_debugging_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_debugging_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::debugging PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_debugging_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_debugging_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_debugging_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_debugging_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::debugging PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_debugging_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_debugging_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_debugging_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_debugging_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_debugging_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_debugging_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_debugging_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_debugging_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_debugging_TARGET_PROPERTIES TRUE)

########## COMPONENT failure_signal_handler TARGET PROPERTIES ######################################

set_property(TARGET absl::failure_signal_handler PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_failure_signal_handler_LINK_LIBS_DEBUG}
                ${absl_failure_signal_handler_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_failure_signal_handler_LINK_LIBS_RELEASE}
                ${absl_failure_signal_handler_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_failure_signal_handler_LINK_LIBS_RELWITHDEBINFO}
                ${absl_failure_signal_handler_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_failure_signal_handler_LINK_LIBS_MINSIZEREL}
                ${absl_failure_signal_handler_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::failure_signal_handler PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_failure_signal_handler_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_failure_signal_handler_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_failure_signal_handler_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_failure_signal_handler_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::failure_signal_handler PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_failure_signal_handler_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_failure_signal_handler_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_failure_signal_handler_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_failure_signal_handler_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::failure_signal_handler PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_failure_signal_handler_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_failure_signal_handler_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_failure_signal_handler_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_failure_signal_handler_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_failure_signal_handler_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_failure_signal_handler_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_failure_signal_handler_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_failure_signal_handler_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_failure_signal_handler_TARGET_PROPERTIES TRUE)

########## COMPONENT node_slot_policy TARGET PROPERTIES ######################################

set_property(TARGET absl::node_slot_policy PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_node_slot_policy_LINK_LIBS_DEBUG}
                ${absl_node_slot_policy_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_node_slot_policy_LINK_LIBS_RELEASE}
                ${absl_node_slot_policy_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_node_slot_policy_LINK_LIBS_RELWITHDEBINFO}
                ${absl_node_slot_policy_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_node_slot_policy_LINK_LIBS_MINSIZEREL}
                ${absl_node_slot_policy_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::node_slot_policy PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_node_slot_policy_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_node_slot_policy_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_node_slot_policy_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_node_slot_policy_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::node_slot_policy PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_node_slot_policy_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_node_slot_policy_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_node_slot_policy_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_node_slot_policy_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::node_slot_policy PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_node_slot_policy_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_node_slot_policy_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_node_slot_policy_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_node_slot_policy_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_node_slot_policy_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_node_slot_policy_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_node_slot_policy_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_node_slot_policy_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_node_slot_policy_TARGET_PROPERTIES TRUE)

########## COMPONENT hashtable_debug TARGET PROPERTIES ######################################

set_property(TARGET absl::hashtable_debug PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_hashtable_debug_LINK_LIBS_DEBUG}
                ${absl_hashtable_debug_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_hashtable_debug_LINK_LIBS_RELEASE}
                ${absl_hashtable_debug_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_hashtable_debug_LINK_LIBS_RELWITHDEBINFO}
                ${absl_hashtable_debug_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_hashtable_debug_LINK_LIBS_MINSIZEREL}
                ${absl_hashtable_debug_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::hashtable_debug PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_hashtable_debug_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_hashtable_debug_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_hashtable_debug_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_hashtable_debug_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::hashtable_debug PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_hashtable_debug_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_hashtable_debug_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_hashtable_debug_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_hashtable_debug_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::hashtable_debug PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_hashtable_debug_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_hashtable_debug_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_hashtable_debug_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_hashtable_debug_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_hashtable_debug_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_hashtable_debug_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_hashtable_debug_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_hashtable_debug_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_hashtable_debug_TARGET_PROPERTIES TRUE)

########## COMPONENT node_hash_set TARGET PROPERTIES ######################################

set_property(TARGET absl::node_hash_set PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_node_hash_set_LINK_LIBS_DEBUG}
                ${absl_node_hash_set_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_node_hash_set_LINK_LIBS_RELEASE}
                ${absl_node_hash_set_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_node_hash_set_LINK_LIBS_RELWITHDEBINFO}
                ${absl_node_hash_set_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_node_hash_set_LINK_LIBS_MINSIZEREL}
                ${absl_node_hash_set_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::node_hash_set PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_node_hash_set_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_node_hash_set_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_node_hash_set_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_node_hash_set_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::node_hash_set PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_node_hash_set_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_node_hash_set_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_node_hash_set_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_node_hash_set_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::node_hash_set PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_node_hash_set_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_node_hash_set_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_node_hash_set_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_node_hash_set_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_node_hash_set_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_node_hash_set_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_node_hash_set_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_node_hash_set_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_node_hash_set_TARGET_PROPERTIES TRUE)

########## COMPONENT node_hash_map TARGET PROPERTIES ######################################

set_property(TARGET absl::node_hash_map PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_node_hash_map_LINK_LIBS_DEBUG}
                ${absl_node_hash_map_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_node_hash_map_LINK_LIBS_RELEASE}
                ${absl_node_hash_map_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_node_hash_map_LINK_LIBS_RELWITHDEBINFO}
                ${absl_node_hash_map_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_node_hash_map_LINK_LIBS_MINSIZEREL}
                ${absl_node_hash_map_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::node_hash_map PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_node_hash_map_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_node_hash_map_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_node_hash_map_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_node_hash_map_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::node_hash_map PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_node_hash_map_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_node_hash_map_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_node_hash_map_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_node_hash_map_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::node_hash_map PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_node_hash_map_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_node_hash_map_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_node_hash_map_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_node_hash_map_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_node_hash_map_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_node_hash_map_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_node_hash_map_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_node_hash_map_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_node_hash_map_TARGET_PROPERTIES TRUE)

########## COMPONENT flat_hash_set TARGET PROPERTIES ######################################

set_property(TARGET absl::flat_hash_set PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_flat_hash_set_LINK_LIBS_DEBUG}
                ${absl_flat_hash_set_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_flat_hash_set_LINK_LIBS_RELEASE}
                ${absl_flat_hash_set_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_flat_hash_set_LINK_LIBS_RELWITHDEBINFO}
                ${absl_flat_hash_set_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_flat_hash_set_LINK_LIBS_MINSIZEREL}
                ${absl_flat_hash_set_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::flat_hash_set PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_flat_hash_set_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_flat_hash_set_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_flat_hash_set_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_flat_hash_set_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::flat_hash_set PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_flat_hash_set_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_flat_hash_set_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_flat_hash_set_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_flat_hash_set_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::flat_hash_set PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_flat_hash_set_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_flat_hash_set_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_flat_hash_set_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_flat_hash_set_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_flat_hash_set_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_flat_hash_set_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_flat_hash_set_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_flat_hash_set_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_flat_hash_set_TARGET_PROPERTIES TRUE)

########## COMPONENT counting_allocator TARGET PROPERTIES ######################################

set_property(TARGET absl::counting_allocator PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_counting_allocator_LINK_LIBS_DEBUG}
                ${absl_counting_allocator_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_counting_allocator_LINK_LIBS_RELEASE}
                ${absl_counting_allocator_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_counting_allocator_LINK_LIBS_RELWITHDEBINFO}
                ${absl_counting_allocator_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_counting_allocator_LINK_LIBS_MINSIZEREL}
                ${absl_counting_allocator_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::counting_allocator PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_counting_allocator_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_counting_allocator_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_counting_allocator_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_counting_allocator_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::counting_allocator PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_counting_allocator_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_counting_allocator_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_counting_allocator_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_counting_allocator_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::counting_allocator PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_counting_allocator_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_counting_allocator_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_counting_allocator_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_counting_allocator_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_counting_allocator_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_counting_allocator_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_counting_allocator_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_counting_allocator_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_counting_allocator_TARGET_PROPERTIES TRUE)

########## COMPONENT btree TARGET PROPERTIES ######################################

set_property(TARGET absl::btree PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_btree_LINK_LIBS_DEBUG}
                ${absl_btree_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_btree_LINK_LIBS_RELEASE}
                ${absl_btree_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_btree_LINK_LIBS_RELWITHDEBINFO}
                ${absl_btree_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_btree_LINK_LIBS_MINSIZEREL}
                ${absl_btree_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::btree PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_btree_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_btree_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_btree_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_btree_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::btree PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_btree_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_btree_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_btree_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_btree_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::btree PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_btree_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_btree_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_btree_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_btree_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_btree_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_btree_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_btree_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_btree_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_btree_TARGET_PROPERTIES TRUE)

########## COMPONENT scoped_set_env TARGET PROPERTIES ######################################

set_property(TARGET absl::scoped_set_env PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_scoped_set_env_LINK_LIBS_DEBUG}
                ${absl_scoped_set_env_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_scoped_set_env_LINK_LIBS_RELEASE}
                ${absl_scoped_set_env_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_scoped_set_env_LINK_LIBS_RELWITHDEBINFO}
                ${absl_scoped_set_env_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_scoped_set_env_LINK_LIBS_MINSIZEREL}
                ${absl_scoped_set_env_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::scoped_set_env PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_scoped_set_env_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_scoped_set_env_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_scoped_set_env_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_scoped_set_env_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::scoped_set_env PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_scoped_set_env_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_scoped_set_env_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_scoped_set_env_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_scoped_set_env_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::scoped_set_env PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_scoped_set_env_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_scoped_set_env_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_scoped_set_env_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_scoped_set_env_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_scoped_set_env_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_scoped_set_env_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_scoped_set_env_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_scoped_set_env_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_scoped_set_env_TARGET_PROPERTIES TRUE)

########## COMPONENT pretty_function TARGET PROPERTIES ######################################

set_property(TARGET absl::pretty_function PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${absl_pretty_function_LINK_LIBS_DEBUG}
                ${absl_pretty_function_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${absl_pretty_function_LINK_LIBS_RELEASE}
                ${absl_pretty_function_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_pretty_function_LINK_LIBS_RELWITHDEBINFO}
                ${absl_pretty_function_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_pretty_function_LINK_LIBS_MINSIZEREL}
                ${absl_pretty_function_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET absl::pretty_function PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${absl_pretty_function_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${absl_pretty_function_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_pretty_function_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_pretty_function_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET absl::pretty_function PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${absl_pretty_function_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${absl_pretty_function_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${absl_pretty_function_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${absl_pretty_function_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET absl::pretty_function PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${absl_pretty_function_COMPILE_OPTIONS_C_DEBUG}
                 ${absl_pretty_function_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${absl_pretty_function_COMPILE_OPTIONS_C_RELEASE}
                 ${absl_pretty_function_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${absl_pretty_function_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${absl_pretty_function_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${absl_pretty_function_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${absl_pretty_function_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(absl_pretty_function_TARGET_PROPERTIES TRUE)

########## GLOBAL TARGET PROPERTIES #########################################################

if(NOT absl_absl_TARGET_PROPERTIES)
    set_property(TARGET absl::absl APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                 $<$<CONFIG:Debug>:${absl_COMPONENTS_DEBUG}>
                 $<$<CONFIG:Release>:${absl_COMPONENTS_RELEASE}>
                 $<$<CONFIG:RelWithDebInfo>:${absl_COMPONENTS_RELWITHDEBINFO}>
                 $<$<CONFIG:MinSizeRel>:${absl_COMPONENTS_MINSIZEREL}>)
endif()

########## BUILD MODULES ####################################################################
#############################################################################################

########## COMPONENT config BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_config_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_config_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_config_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_config_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT type_traits BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_type_traits_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_type_traits_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_type_traits_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_type_traits_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT base_internal BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_base_internal_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_base_internal_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_base_internal_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_base_internal_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT utility BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_utility_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_utility_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_utility_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_utility_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT core_headers BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_core_headers_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_core_headers_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_core_headers_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_core_headers_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT compare BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_compare_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_compare_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_compare_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_compare_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT log_severity BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_log_severity_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_log_severity_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_log_severity_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_log_severity_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT errno_saver BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_errno_saver_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_errno_saver_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_errno_saver_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_errno_saver_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT atomic_hook BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_atomic_hook_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_atomic_hook_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_atomic_hook_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_atomic_hook_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT raw_logging_internal BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_raw_logging_internal_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_raw_logging_internal_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_raw_logging_internal_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_raw_logging_internal_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT bad_variant_access BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_bad_variant_access_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_bad_variant_access_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_bad_variant_access_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_bad_variant_access_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT variant BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_variant_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_variant_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_variant_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_variant_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT bad_optional_access BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_bad_optional_access_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_bad_optional_access_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_bad_optional_access_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_bad_optional_access_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT meta BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_meta_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_meta_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_meta_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_meta_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT memory BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_memory_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_memory_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_memory_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_memory_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT optional BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_optional_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_optional_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_optional_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_optional_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT algorithm BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_algorithm_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_algorithm_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_algorithm_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_algorithm_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT throw_delegate BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_throw_delegate_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_throw_delegate_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_throw_delegate_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_throw_delegate_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT span BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_span_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_span_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_span_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_span_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT bad_any_cast_impl BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_bad_any_cast_impl_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_bad_any_cast_impl_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_bad_any_cast_impl_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_bad_any_cast_impl_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT bad_any_cast BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_bad_any_cast_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_bad_any_cast_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_bad_any_cast_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_bad_any_cast_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT fast_type_id BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_fast_type_id_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_fast_type_id_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_fast_type_id_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_fast_type_id_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT any BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_any_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_any_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_any_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_any_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT time_zone BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_time_zone_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_time_zone_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_time_zone_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_time_zone_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT civil_time BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_civil_time_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_civil_time_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_civil_time_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_civil_time_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT dynamic_annotations BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_dynamic_annotations_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_dynamic_annotations_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_dynamic_annotations_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_dynamic_annotations_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT spinlock_wait BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_spinlock_wait_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_spinlock_wait_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_spinlock_wait_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_spinlock_wait_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT base BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_base_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_base_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_base_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_base_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT endian BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_endian_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_endian_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_endian_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_endian_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT strings_internal BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_strings_internal_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_strings_internal_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_strings_internal_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_strings_internal_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT bits BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_bits_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_bits_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_bits_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_bits_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT int128 BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_int128_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_int128_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_int128_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_int128_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT strings BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_strings_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_strings_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_strings_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_strings_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT time BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_time_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_time_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_time_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_time_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT kernel_timeout_internal BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_kernel_timeout_internal_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_kernel_timeout_internal_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_kernel_timeout_internal_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_kernel_timeout_internal_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT malloc_internal BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_malloc_internal_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_malloc_internal_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_malloc_internal_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_malloc_internal_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT graphcycles_internal BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_graphcycles_internal_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_graphcycles_internal_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_graphcycles_internal_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_graphcycles_internal_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT demangle_internal BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_demangle_internal_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_demangle_internal_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_demangle_internal_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_demangle_internal_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT debugging_internal BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_debugging_internal_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_debugging_internal_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_debugging_internal_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_debugging_internal_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT symbolize BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_symbolize_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_symbolize_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_symbolize_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_symbolize_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT stacktrace BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_stacktrace_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_stacktrace_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_stacktrace_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_stacktrace_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT synchronization BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_synchronization_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_synchronization_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_synchronization_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_synchronization_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT cordz_handle BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_cordz_handle_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_cordz_handle_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_cordz_handle_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_cordz_handle_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT cordz_update_tracker BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_cordz_update_tracker_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_cordz_update_tracker_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_cordz_update_tracker_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_cordz_update_tracker_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT cordz_statistics BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_cordz_statistics_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_cordz_statistics_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_cordz_statistics_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_cordz_statistics_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT exponential_biased BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_exponential_biased_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_exponential_biased_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_exponential_biased_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_exponential_biased_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT cordz_functions BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_cordz_functions_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_cordz_functions_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_cordz_functions_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_cordz_functions_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT non_temporal_arm_intrinsics BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_non_temporal_arm_intrinsics_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_non_temporal_arm_intrinsics_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_non_temporal_arm_intrinsics_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_non_temporal_arm_intrinsics_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT non_temporal_memcpy BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_non_temporal_memcpy_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_non_temporal_memcpy_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_non_temporal_memcpy_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_non_temporal_memcpy_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT crc_cpu_detect BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_crc_cpu_detect_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_crc_cpu_detect_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_crc_cpu_detect_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_crc_cpu_detect_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT prefetch BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_prefetch_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_prefetch_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_prefetch_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_prefetch_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT crc_internal BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_crc_internal_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_crc_internal_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_crc_internal_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_crc_internal_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT crc32c BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_crc32c_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_crc32c_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_crc32c_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_crc32c_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT crc_cord_state BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_crc_cord_state_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_crc_cord_state_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_crc_cord_state_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_crc_cord_state_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT layout BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_layout_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_layout_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_layout_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_layout_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT container_memory BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_container_memory_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_container_memory_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_container_memory_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_container_memory_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT compressed_tuple BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_compressed_tuple_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_compressed_tuple_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_compressed_tuple_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_compressed_tuple_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT inlined_vector_internal BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_inlined_vector_internal_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_inlined_vector_internal_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_inlined_vector_internal_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_inlined_vector_internal_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT inlined_vector BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_inlined_vector_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_inlined_vector_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_inlined_vector_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_inlined_vector_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT cord_internal BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_cord_internal_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_cord_internal_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_cord_internal_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_cord_internal_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT cordz_info BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_cordz_info_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_cordz_info_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_cordz_info_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_cordz_info_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT cordz_update_scope BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_cordz_update_scope_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_cordz_update_scope_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_cordz_update_scope_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_cordz_update_scope_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT function_ref BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_function_ref_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_function_ref_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_function_ref_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_function_ref_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT fixed_array BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_fixed_array_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_fixed_array_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_fixed_array_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_fixed_array_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT cord BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_cord_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_cord_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_cord_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_cord_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT cordz_sample_token BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_cordz_sample_token_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_cordz_sample_token_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_cordz_sample_token_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_cordz_sample_token_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT numeric_representation BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_numeric_representation_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_numeric_representation_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_numeric_representation_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_numeric_representation_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT str_format_internal BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_str_format_internal_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_str_format_internal_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_str_format_internal_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_str_format_internal_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT str_format BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_str_format_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_str_format_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_str_format_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_str_format_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT strerror BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_strerror_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_strerror_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_strerror_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_strerror_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT status BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_status_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_status_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_status_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_status_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT statusor BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_statusor_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_statusor_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_statusor_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_statusor_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT random_internal_traits BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_random_internal_traits_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_random_internal_traits_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_random_internal_traits_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_random_internal_traits_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT random_internal_uniform_helper BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_random_internal_uniform_helper_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_random_internal_uniform_helper_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_random_internal_uniform_helper_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_random_internal_uniform_helper_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT random_internal_distribution_test_util BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_random_internal_distribution_test_util_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_random_internal_distribution_test_util_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_random_internal_distribution_test_util_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_random_internal_distribution_test_util_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT random_internal_platform BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_random_internal_platform_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_random_internal_platform_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_random_internal_platform_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_random_internal_platform_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT random_internal_randen_hwaes_impl BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_random_internal_randen_hwaes_impl_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_random_internal_randen_hwaes_impl_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_random_internal_randen_hwaes_impl_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_random_internal_randen_hwaes_impl_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT random_internal_randen_hwaes BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_random_internal_randen_hwaes_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_random_internal_randen_hwaes_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_random_internal_randen_hwaes_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_random_internal_randen_hwaes_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT random_internal_randen_slow BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_random_internal_randen_slow_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_random_internal_randen_slow_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_random_internal_randen_slow_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_random_internal_randen_slow_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT random_internal_randen BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_random_internal_randen_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_random_internal_randen_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_random_internal_randen_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_random_internal_randen_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT random_internal_iostream_state_saver BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_random_internal_iostream_state_saver_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_random_internal_iostream_state_saver_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_random_internal_iostream_state_saver_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_random_internal_iostream_state_saver_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT random_internal_randen_engine BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_random_internal_randen_engine_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_random_internal_randen_engine_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_random_internal_randen_engine_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_random_internal_randen_engine_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT random_internal_fastmath BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_random_internal_fastmath_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_random_internal_fastmath_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_random_internal_fastmath_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_random_internal_fastmath_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT random_internal_pcg_engine BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_random_internal_pcg_engine_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_random_internal_pcg_engine_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_random_internal_pcg_engine_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_random_internal_pcg_engine_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT random_internal_fast_uniform_bits BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_random_internal_fast_uniform_bits_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_random_internal_fast_uniform_bits_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_random_internal_fast_uniform_bits_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_random_internal_fast_uniform_bits_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT random_internal_seed_material BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_random_internal_seed_material_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_random_internal_seed_material_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_random_internal_seed_material_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_random_internal_seed_material_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT random_internal_salted_seed_seq BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_random_internal_salted_seed_seq_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_random_internal_salted_seed_seq_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_random_internal_salted_seed_seq_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_random_internal_salted_seed_seq_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT random_seed_gen_exception BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_random_seed_gen_exception_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_random_seed_gen_exception_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_random_seed_gen_exception_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_random_seed_gen_exception_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT random_internal_pool_urbg BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_random_internal_pool_urbg_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_random_internal_pool_urbg_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_random_internal_pool_urbg_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_random_internal_pool_urbg_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT random_internal_nonsecure_base BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_random_internal_nonsecure_base_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_random_internal_nonsecure_base_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_random_internal_nonsecure_base_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_random_internal_nonsecure_base_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT random_internal_wide_multiply BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_random_internal_wide_multiply_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_random_internal_wide_multiply_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_random_internal_wide_multiply_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_random_internal_wide_multiply_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT random_internal_generate_real BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_random_internal_generate_real_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_random_internal_generate_real_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_random_internal_generate_real_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_random_internal_generate_real_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT random_internal_distribution_caller BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_random_internal_distribution_caller_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_random_internal_distribution_caller_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_random_internal_distribution_caller_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_random_internal_distribution_caller_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT random_seed_sequences BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_random_seed_sequences_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_random_seed_sequences_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_random_seed_sequences_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_random_seed_sequences_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT random_distributions BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_random_distributions_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_random_distributions_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_random_distributions_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_random_distributions_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT random_internal_mock_helpers BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_random_internal_mock_helpers_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_random_internal_mock_helpers_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_random_internal_mock_helpers_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_random_internal_mock_helpers_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT random_bit_gen_ref BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_random_bit_gen_ref_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_random_bit_gen_ref_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_random_bit_gen_ref_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_random_bit_gen_ref_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT random_random BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_random_random_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_random_random_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_random_random_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_random_random_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT periodic_sampler BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_periodic_sampler_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_periodic_sampler_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_periodic_sampler_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_periodic_sampler_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT sample_recorder BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_sample_recorder_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_sample_recorder_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_sample_recorder_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_sample_recorder_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT numeric BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_numeric_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_numeric_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_numeric_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_numeric_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT log_internal_config BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_log_internal_config_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_log_internal_config_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_log_internal_config_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_log_internal_config_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT log_entry BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_log_entry_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_log_entry_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_log_entry_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_log_entry_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT log_sink BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_log_sink_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_log_sink_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_log_sink_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_log_sink_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT low_level_hash BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_low_level_hash_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_low_level_hash_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_low_level_hash_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_low_level_hash_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT city BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_city_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_city_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_city_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_city_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT hash BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_hash_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_hash_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_hash_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_hash_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT log_globals BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_log_globals_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_log_globals_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_log_globals_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_log_globals_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT log_internal_globals BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_log_internal_globals_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_log_internal_globals_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_log_internal_globals_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_log_internal_globals_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT cleanup_internal BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_cleanup_internal_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_cleanup_internal_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_cleanup_internal_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_cleanup_internal_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT cleanup BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_cleanup_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_cleanup_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_cleanup_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_cleanup_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT log_internal_log_sink_set BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_log_internal_log_sink_set_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_log_internal_log_sink_set_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_log_internal_log_sink_set_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_log_internal_log_sink_set_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT log_sink_registry BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_log_sink_registry_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_log_sink_registry_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_log_sink_registry_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_log_sink_registry_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT log_internal_append_truncated BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_log_internal_append_truncated_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_log_internal_append_truncated_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_log_internal_append_truncated_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_log_internal_append_truncated_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT log_internal_nullguard BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_log_internal_nullguard_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_log_internal_nullguard_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_log_internal_nullguard_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_log_internal_nullguard_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT log_internal_proto BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_log_internal_proto_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_log_internal_proto_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_log_internal_proto_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_log_internal_proto_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT log_internal_format BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_log_internal_format_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_log_internal_format_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_log_internal_format_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_log_internal_format_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT examine_stack BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_examine_stack_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_examine_stack_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_examine_stack_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_examine_stack_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT log_internal_message BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_log_internal_message_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_log_internal_message_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_log_internal_message_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_log_internal_message_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT log_internal_structured BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_log_internal_structured_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_log_internal_structured_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_log_internal_structured_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_log_internal_structured_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT log_structured BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_log_structured_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_log_structured_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_log_structured_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_log_structured_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT log_internal_voidify BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_log_internal_voidify_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_log_internal_voidify_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_log_internal_voidify_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_log_internal_voidify_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT log_internal_nullstream BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_log_internal_nullstream_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_log_internal_nullstream_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_log_internal_nullstream_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_log_internal_nullstream_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT log_internal_strip BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_log_internal_strip_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_log_internal_strip_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_log_internal_strip_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_log_internal_strip_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT log_internal_conditions BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_log_internal_conditions_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_log_internal_conditions_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_log_internal_conditions_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_log_internal_conditions_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT log_internal_log_impl BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_log_internal_log_impl_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_log_internal_log_impl_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_log_internal_log_impl_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_log_internal_log_impl_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT absl_log BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_absl_log_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_absl_log_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_absl_log_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_absl_log_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT log_streamer BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_log_streamer_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_log_streamer_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_log_streamer_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_log_streamer_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT log BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_log_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_log_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_log_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_log_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT log_initialize BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_log_initialize_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_log_initialize_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_log_initialize_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_log_initialize_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT flags_commandlineflag_internal BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_flags_commandlineflag_internal_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_flags_commandlineflag_internal_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_flags_commandlineflag_internal_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_flags_commandlineflag_internal_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT flags_commandlineflag BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_flags_commandlineflag_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_flags_commandlineflag_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_flags_commandlineflag_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_flags_commandlineflag_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT flags_marshalling BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_flags_marshalling_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_flags_marshalling_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_flags_marshalling_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_flags_marshalling_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT flags_path_util BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_flags_path_util_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_flags_path_util_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_flags_path_util_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_flags_path_util_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT flags_program_name BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_flags_program_name_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_flags_program_name_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_flags_program_name_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_flags_program_name_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT flags_config BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_flags_config_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_flags_config_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_flags_config_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_flags_config_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT flags_internal BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_flags_internal_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_flags_internal_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_flags_internal_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_flags_internal_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT flags_private_handle_accessor BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_flags_private_handle_accessor_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_flags_private_handle_accessor_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_flags_private_handle_accessor_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_flags_private_handle_accessor_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT container_common BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_container_common_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_container_common_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_container_common_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_container_common_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT hashtable_debug_hooks BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_hashtable_debug_hooks_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_hashtable_debug_hooks_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_hashtable_debug_hooks_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_hashtable_debug_hooks_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT hashtablez_sampler BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_hashtablez_sampler_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_hashtablez_sampler_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_hashtablez_sampler_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_hashtablez_sampler_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT common_policy_traits BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_common_policy_traits_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_common_policy_traits_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_common_policy_traits_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_common_policy_traits_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT hash_policy_traits BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_hash_policy_traits_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_hash_policy_traits_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_hash_policy_traits_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_hash_policy_traits_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT raw_hash_set BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_raw_hash_set_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_raw_hash_set_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_raw_hash_set_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_raw_hash_set_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT raw_hash_map BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_raw_hash_map_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_raw_hash_map_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_raw_hash_map_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_raw_hash_map_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT hash_function_defaults BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_hash_function_defaults_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_hash_function_defaults_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_hash_function_defaults_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_hash_function_defaults_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT algorithm_container BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_algorithm_container_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_algorithm_container_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_algorithm_container_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_algorithm_container_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT flat_hash_map BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_flat_hash_map_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_flat_hash_map_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_flat_hash_map_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_flat_hash_map_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT flags_reflection BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_flags_reflection_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_flags_reflection_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_flags_reflection_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_flags_reflection_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT flags BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_flags_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_flags_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_flags_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_flags_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT log_internal_flags BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_log_internal_flags_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_log_internal_flags_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_log_internal_flags_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_log_internal_flags_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT log_flags BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_log_flags_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_log_flags_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_log_flags_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_log_flags_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT die_if_null BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_die_if_null_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_die_if_null_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_die_if_null_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_die_if_null_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT log_internal_check_op BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_log_internal_check_op_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_log_internal_check_op_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_log_internal_check_op_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_log_internal_check_op_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT log_internal_check_impl BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_log_internal_check_impl_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_log_internal_check_impl_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_log_internal_check_impl_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_log_internal_check_impl_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT check BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_check_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_check_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_check_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_check_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT absl_check BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_absl_check_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_absl_check_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_absl_check_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_absl_check_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT bind_front BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_bind_front_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_bind_front_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_bind_front_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_bind_front_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT any_invocable BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_any_invocable_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_any_invocable_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_any_invocable_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_any_invocable_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT flags_usage_internal BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_flags_usage_internal_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_flags_usage_internal_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_flags_usage_internal_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_flags_usage_internal_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT flags_usage BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_flags_usage_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_flags_usage_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_flags_usage_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_flags_usage_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT flags_parse BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_flags_parse_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_flags_parse_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_flags_parse_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_flags_parse_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT leak_check BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_leak_check_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_leak_check_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_leak_check_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_leak_check_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT debugging BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_debugging_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_debugging_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_debugging_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_debugging_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT failure_signal_handler BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_failure_signal_handler_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_failure_signal_handler_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_failure_signal_handler_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_failure_signal_handler_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT node_slot_policy BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_node_slot_policy_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_node_slot_policy_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_node_slot_policy_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_node_slot_policy_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT hashtable_debug BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_hashtable_debug_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_hashtable_debug_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_hashtable_debug_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_hashtable_debug_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT node_hash_set BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_node_hash_set_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_node_hash_set_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_node_hash_set_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_node_hash_set_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT node_hash_map BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_node_hash_map_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_node_hash_map_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_node_hash_map_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_node_hash_map_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT flat_hash_set BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_flat_hash_set_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_flat_hash_set_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_flat_hash_set_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_flat_hash_set_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT counting_allocator BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_counting_allocator_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_counting_allocator_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_counting_allocator_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_counting_allocator_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT btree BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_btree_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_btree_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_btree_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_btree_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT scoped_set_env BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_scoped_set_env_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_scoped_set_env_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_scoped_set_env_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_scoped_set_env_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT pretty_function BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${absl_pretty_function_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_pretty_function_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_pretty_function_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${absl_pretty_function_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()