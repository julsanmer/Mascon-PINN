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

include(${CMAKE_CURRENT_LIST_DIR}/BoostTargets.cmake)

########## FIND PACKAGE DEPENDENCY ##########################################################
#############################################################################################

include(CMakeFindDependencyMacro)

if(NOT BZip2_FOUND)
    if(${CMAKE_VERSION} VERSION_LESS "3.9.0")
        find_package(BZip2 REQUIRED NO_MODULE)
    else()
        find_dependency(BZip2 REQUIRED NO_MODULE)
    endif()
else()
    message(STATUS "Dependency BZip2 already found")
endif()

if(NOT ZLIB_FOUND)
    if(${CMAKE_VERSION} VERSION_LESS "3.9.0")
        find_package(ZLIB REQUIRED NO_MODULE)
    else()
        find_dependency(ZLIB REQUIRED NO_MODULE)
    endif()
else()
    message(STATUS "Dependency ZLIB already found")
endif()

if(NOT libbacktrace_FOUND)
    if(${CMAKE_VERSION} VERSION_LESS "3.9.0")
        find_package(libbacktrace REQUIRED NO_MODULE)
    else()
        find_dependency(libbacktrace REQUIRED NO_MODULE)
    endif()
else()
    message(STATUS "Dependency libbacktrace already found")
endif()

########## TARGETS PROPERTIES ###############################################################
#############################################################################################

########## COMPONENT dynamic_linking TARGET PROPERTIES ######################################

set_property(TARGET Boost::dynamic_linking PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${Boost_dynamic_linking_LINK_LIBS_DEBUG}
                ${Boost_dynamic_linking_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${Boost_dynamic_linking_LINK_LIBS_RELEASE}
                ${Boost_dynamic_linking_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_dynamic_linking_LINK_LIBS_RELWITHDEBINFO}
                ${Boost_dynamic_linking_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_dynamic_linking_LINK_LIBS_MINSIZEREL}
                ${Boost_dynamic_linking_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET Boost::dynamic_linking PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${Boost_dynamic_linking_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${Boost_dynamic_linking_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_dynamic_linking_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_dynamic_linking_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET Boost::dynamic_linking PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${Boost_dynamic_linking_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${Boost_dynamic_linking_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_dynamic_linking_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_dynamic_linking_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET Boost::dynamic_linking PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${Boost_dynamic_linking_COMPILE_OPTIONS_C_DEBUG}
                 ${Boost_dynamic_linking_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${Boost_dynamic_linking_COMPILE_OPTIONS_C_RELEASE}
                 ${Boost_dynamic_linking_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${Boost_dynamic_linking_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${Boost_dynamic_linking_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${Boost_dynamic_linking_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${Boost_dynamic_linking_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(Boost_dynamic_linking_TARGET_PROPERTIES TRUE)

########## COMPONENT disable_autolinking TARGET PROPERTIES ######################################

set_property(TARGET Boost::disable_autolinking PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${Boost_disable_autolinking_LINK_LIBS_DEBUG}
                ${Boost_disable_autolinking_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${Boost_disable_autolinking_LINK_LIBS_RELEASE}
                ${Boost_disable_autolinking_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_disable_autolinking_LINK_LIBS_RELWITHDEBINFO}
                ${Boost_disable_autolinking_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_disable_autolinking_LINK_LIBS_MINSIZEREL}
                ${Boost_disable_autolinking_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET Boost::disable_autolinking PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${Boost_disable_autolinking_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${Boost_disable_autolinking_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_disable_autolinking_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_disable_autolinking_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET Boost::disable_autolinking PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${Boost_disable_autolinking_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${Boost_disable_autolinking_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_disable_autolinking_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_disable_autolinking_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET Boost::disable_autolinking PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${Boost_disable_autolinking_COMPILE_OPTIONS_C_DEBUG}
                 ${Boost_disable_autolinking_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${Boost_disable_autolinking_COMPILE_OPTIONS_C_RELEASE}
                 ${Boost_disable_autolinking_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${Boost_disable_autolinking_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${Boost_disable_autolinking_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${Boost_disable_autolinking_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${Boost_disable_autolinking_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(Boost_disable_autolinking_TARGET_PROPERTIES TRUE)

########## COMPONENT diagnostic_definitions TARGET PROPERTIES ######################################

set_property(TARGET Boost::diagnostic_definitions PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${Boost_diagnostic_definitions_LINK_LIBS_DEBUG}
                ${Boost_diagnostic_definitions_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${Boost_diagnostic_definitions_LINK_LIBS_RELEASE}
                ${Boost_diagnostic_definitions_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_diagnostic_definitions_LINK_LIBS_RELWITHDEBINFO}
                ${Boost_diagnostic_definitions_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_diagnostic_definitions_LINK_LIBS_MINSIZEREL}
                ${Boost_diagnostic_definitions_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET Boost::diagnostic_definitions PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${Boost_diagnostic_definitions_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${Boost_diagnostic_definitions_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_diagnostic_definitions_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_diagnostic_definitions_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET Boost::diagnostic_definitions PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${Boost_diagnostic_definitions_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${Boost_diagnostic_definitions_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_diagnostic_definitions_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_diagnostic_definitions_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET Boost::diagnostic_definitions PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${Boost_diagnostic_definitions_COMPILE_OPTIONS_C_DEBUG}
                 ${Boost_diagnostic_definitions_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${Boost_diagnostic_definitions_COMPILE_OPTIONS_C_RELEASE}
                 ${Boost_diagnostic_definitions_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${Boost_diagnostic_definitions_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${Boost_diagnostic_definitions_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${Boost_diagnostic_definitions_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${Boost_diagnostic_definitions_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(Boost_diagnostic_definitions_TARGET_PROPERTIES TRUE)

########## COMPONENT headers TARGET PROPERTIES ######################################

set_property(TARGET Boost::headers PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${Boost_headers_LINK_LIBS_DEBUG}
                ${Boost_headers_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${Boost_headers_LINK_LIBS_RELEASE}
                ${Boost_headers_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_headers_LINK_LIBS_RELWITHDEBINFO}
                ${Boost_headers_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_headers_LINK_LIBS_MINSIZEREL}
                ${Boost_headers_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET Boost::headers PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${Boost_headers_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${Boost_headers_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_headers_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_headers_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET Boost::headers PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${Boost_headers_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${Boost_headers_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_headers_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_headers_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET Boost::headers PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${Boost_headers_COMPILE_OPTIONS_C_DEBUG}
                 ${Boost_headers_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${Boost_headers_COMPILE_OPTIONS_C_RELEASE}
                 ${Boost_headers_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${Boost_headers_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${Boost_headers_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${Boost_headers_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${Boost_headers_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(Boost_headers_TARGET_PROPERTIES TRUE)

########## COMPONENT _libboost TARGET PROPERTIES ######################################

set_property(TARGET Boost::_libboost PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${Boost__libboost_LINK_LIBS_DEBUG}
                ${Boost__libboost_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${Boost__libboost_LINK_LIBS_RELEASE}
                ${Boost__libboost_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost__libboost_LINK_LIBS_RELWITHDEBINFO}
                ${Boost__libboost_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost__libboost_LINK_LIBS_MINSIZEREL}
                ${Boost__libboost_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET Boost::_libboost PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${Boost__libboost_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${Boost__libboost_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost__libboost_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost__libboost_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET Boost::_libboost PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${Boost__libboost_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${Boost__libboost_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost__libboost_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost__libboost_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET Boost::_libboost PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${Boost__libboost_COMPILE_OPTIONS_C_DEBUG}
                 ${Boost__libboost_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${Boost__libboost_COMPILE_OPTIONS_C_RELEASE}
                 ${Boost__libboost_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${Boost__libboost_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${Boost__libboost_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${Boost__libboost_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${Boost__libboost_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(Boost__libboost_TARGET_PROPERTIES TRUE)

########## COMPONENT serialization TARGET PROPERTIES ######################################

set_property(TARGET Boost::serialization PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${Boost_serialization_LINK_LIBS_DEBUG}
                ${Boost_serialization_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${Boost_serialization_LINK_LIBS_RELEASE}
                ${Boost_serialization_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_serialization_LINK_LIBS_RELWITHDEBINFO}
                ${Boost_serialization_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_serialization_LINK_LIBS_MINSIZEREL}
                ${Boost_serialization_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET Boost::serialization PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${Boost_serialization_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${Boost_serialization_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_serialization_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_serialization_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET Boost::serialization PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${Boost_serialization_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${Boost_serialization_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_serialization_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_serialization_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET Boost::serialization PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${Boost_serialization_COMPILE_OPTIONS_C_DEBUG}
                 ${Boost_serialization_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${Boost_serialization_COMPILE_OPTIONS_C_RELEASE}
                 ${Boost_serialization_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${Boost_serialization_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${Boost_serialization_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${Boost_serialization_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${Boost_serialization_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(Boost_serialization_TARGET_PROPERTIES TRUE)

########## COMPONENT wserialization TARGET PROPERTIES ######################################

set_property(TARGET Boost::wserialization PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${Boost_wserialization_LINK_LIBS_DEBUG}
                ${Boost_wserialization_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${Boost_wserialization_LINK_LIBS_RELEASE}
                ${Boost_wserialization_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_wserialization_LINK_LIBS_RELWITHDEBINFO}
                ${Boost_wserialization_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_wserialization_LINK_LIBS_MINSIZEREL}
                ${Boost_wserialization_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET Boost::wserialization PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${Boost_wserialization_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${Boost_wserialization_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_wserialization_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_wserialization_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET Boost::wserialization PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${Boost_wserialization_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${Boost_wserialization_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_wserialization_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_wserialization_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET Boost::wserialization PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${Boost_wserialization_COMPILE_OPTIONS_C_DEBUG}
                 ${Boost_wserialization_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${Boost_wserialization_COMPILE_OPTIONS_C_RELEASE}
                 ${Boost_wserialization_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${Boost_wserialization_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${Boost_wserialization_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${Boost_wserialization_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${Boost_wserialization_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(Boost_wserialization_TARGET_PROPERTIES TRUE)

########## COMPONENT exception TARGET PROPERTIES ######################################

set_property(TARGET Boost::exception PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${Boost_exception_LINK_LIBS_DEBUG}
                ${Boost_exception_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${Boost_exception_LINK_LIBS_RELEASE}
                ${Boost_exception_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_exception_LINK_LIBS_RELWITHDEBINFO}
                ${Boost_exception_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_exception_LINK_LIBS_MINSIZEREL}
                ${Boost_exception_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET Boost::exception PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${Boost_exception_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${Boost_exception_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_exception_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_exception_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET Boost::exception PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${Boost_exception_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${Boost_exception_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_exception_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_exception_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET Boost::exception PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${Boost_exception_COMPILE_OPTIONS_C_DEBUG}
                 ${Boost_exception_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${Boost_exception_COMPILE_OPTIONS_C_RELEASE}
                 ${Boost_exception_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${Boost_exception_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${Boost_exception_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${Boost_exception_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${Boost_exception_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(Boost_exception_TARGET_PROPERTIES TRUE)

########## COMPONENT test TARGET PROPERTIES ######################################

set_property(TARGET Boost::test PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${Boost_test_LINK_LIBS_DEBUG}
                ${Boost_test_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${Boost_test_LINK_LIBS_RELEASE}
                ${Boost_test_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_test_LINK_LIBS_RELWITHDEBINFO}
                ${Boost_test_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_test_LINK_LIBS_MINSIZEREL}
                ${Boost_test_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET Boost::test PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${Boost_test_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${Boost_test_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_test_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_test_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET Boost::test PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${Boost_test_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${Boost_test_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_test_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_test_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET Boost::test PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${Boost_test_COMPILE_OPTIONS_C_DEBUG}
                 ${Boost_test_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${Boost_test_COMPILE_OPTIONS_C_RELEASE}
                 ${Boost_test_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${Boost_test_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${Boost_test_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${Boost_test_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${Boost_test_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(Boost_test_TARGET_PROPERTIES TRUE)

########## COMPONENT test_exec_monitor TARGET PROPERTIES ######################################

set_property(TARGET Boost::test_exec_monitor PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${Boost_test_exec_monitor_LINK_LIBS_DEBUG}
                ${Boost_test_exec_monitor_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${Boost_test_exec_monitor_LINK_LIBS_RELEASE}
                ${Boost_test_exec_monitor_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_test_exec_monitor_LINK_LIBS_RELWITHDEBINFO}
                ${Boost_test_exec_monitor_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_test_exec_monitor_LINK_LIBS_MINSIZEREL}
                ${Boost_test_exec_monitor_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET Boost::test_exec_monitor PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${Boost_test_exec_monitor_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${Boost_test_exec_monitor_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_test_exec_monitor_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_test_exec_monitor_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET Boost::test_exec_monitor PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${Boost_test_exec_monitor_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${Boost_test_exec_monitor_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_test_exec_monitor_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_test_exec_monitor_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET Boost::test_exec_monitor PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${Boost_test_exec_monitor_COMPILE_OPTIONS_C_DEBUG}
                 ${Boost_test_exec_monitor_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${Boost_test_exec_monitor_COMPILE_OPTIONS_C_RELEASE}
                 ${Boost_test_exec_monitor_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${Boost_test_exec_monitor_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${Boost_test_exec_monitor_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${Boost_test_exec_monitor_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${Boost_test_exec_monitor_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(Boost_test_exec_monitor_TARGET_PROPERTIES TRUE)

########## COMPONENT prg_exec_monitor TARGET PROPERTIES ######################################

set_property(TARGET Boost::prg_exec_monitor PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${Boost_prg_exec_monitor_LINK_LIBS_DEBUG}
                ${Boost_prg_exec_monitor_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${Boost_prg_exec_monitor_LINK_LIBS_RELEASE}
                ${Boost_prg_exec_monitor_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_prg_exec_monitor_LINK_LIBS_RELWITHDEBINFO}
                ${Boost_prg_exec_monitor_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_prg_exec_monitor_LINK_LIBS_MINSIZEREL}
                ${Boost_prg_exec_monitor_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET Boost::prg_exec_monitor PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${Boost_prg_exec_monitor_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${Boost_prg_exec_monitor_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_prg_exec_monitor_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_prg_exec_monitor_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET Boost::prg_exec_monitor PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${Boost_prg_exec_monitor_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${Boost_prg_exec_monitor_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_prg_exec_monitor_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_prg_exec_monitor_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET Boost::prg_exec_monitor PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${Boost_prg_exec_monitor_COMPILE_OPTIONS_C_DEBUG}
                 ${Boost_prg_exec_monitor_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${Boost_prg_exec_monitor_COMPILE_OPTIONS_C_RELEASE}
                 ${Boost_prg_exec_monitor_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${Boost_prg_exec_monitor_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${Boost_prg_exec_monitor_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${Boost_prg_exec_monitor_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${Boost_prg_exec_monitor_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(Boost_prg_exec_monitor_TARGET_PROPERTIES TRUE)

########## COMPONENT unit_test_framework TARGET PROPERTIES ######################################

set_property(TARGET Boost::unit_test_framework PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${Boost_unit_test_framework_LINK_LIBS_DEBUG}
                ${Boost_unit_test_framework_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${Boost_unit_test_framework_LINK_LIBS_RELEASE}
                ${Boost_unit_test_framework_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_unit_test_framework_LINK_LIBS_RELWITHDEBINFO}
                ${Boost_unit_test_framework_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_unit_test_framework_LINK_LIBS_MINSIZEREL}
                ${Boost_unit_test_framework_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET Boost::unit_test_framework PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${Boost_unit_test_framework_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${Boost_unit_test_framework_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_unit_test_framework_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_unit_test_framework_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET Boost::unit_test_framework PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${Boost_unit_test_framework_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${Boost_unit_test_framework_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_unit_test_framework_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_unit_test_framework_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET Boost::unit_test_framework PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${Boost_unit_test_framework_COMPILE_OPTIONS_C_DEBUG}
                 ${Boost_unit_test_framework_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${Boost_unit_test_framework_COMPILE_OPTIONS_C_RELEASE}
                 ${Boost_unit_test_framework_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${Boost_unit_test_framework_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${Boost_unit_test_framework_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${Boost_unit_test_framework_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${Boost_unit_test_framework_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(Boost_unit_test_framework_TARGET_PROPERTIES TRUE)

########## COMPONENT system TARGET PROPERTIES ######################################

set_property(TARGET Boost::system PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${Boost_system_LINK_LIBS_DEBUG}
                ${Boost_system_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${Boost_system_LINK_LIBS_RELEASE}
                ${Boost_system_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_system_LINK_LIBS_RELWITHDEBINFO}
                ${Boost_system_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_system_LINK_LIBS_MINSIZEREL}
                ${Boost_system_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET Boost::system PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${Boost_system_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${Boost_system_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_system_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_system_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET Boost::system PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${Boost_system_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${Boost_system_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_system_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_system_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET Boost::system PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${Boost_system_COMPILE_OPTIONS_C_DEBUG}
                 ${Boost_system_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${Boost_system_COMPILE_OPTIONS_C_RELEASE}
                 ${Boost_system_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${Boost_system_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${Boost_system_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${Boost_system_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${Boost_system_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(Boost_system_TARGET_PROPERTIES TRUE)

########## COMPONENT date_time TARGET PROPERTIES ######################################

set_property(TARGET Boost::date_time PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${Boost_date_time_LINK_LIBS_DEBUG}
                ${Boost_date_time_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${Boost_date_time_LINK_LIBS_RELEASE}
                ${Boost_date_time_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_date_time_LINK_LIBS_RELWITHDEBINFO}
                ${Boost_date_time_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_date_time_LINK_LIBS_MINSIZEREL}
                ${Boost_date_time_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET Boost::date_time PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${Boost_date_time_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${Boost_date_time_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_date_time_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_date_time_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET Boost::date_time PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${Boost_date_time_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${Boost_date_time_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_date_time_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_date_time_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET Boost::date_time PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${Boost_date_time_COMPILE_OPTIONS_C_DEBUG}
                 ${Boost_date_time_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${Boost_date_time_COMPILE_OPTIONS_C_RELEASE}
                 ${Boost_date_time_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${Boost_date_time_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${Boost_date_time_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${Boost_date_time_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${Boost_date_time_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(Boost_date_time_TARGET_PROPERTIES TRUE)

########## COMPONENT container TARGET PROPERTIES ######################################

set_property(TARGET Boost::container PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${Boost_container_LINK_LIBS_DEBUG}
                ${Boost_container_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${Boost_container_LINK_LIBS_RELEASE}
                ${Boost_container_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_container_LINK_LIBS_RELWITHDEBINFO}
                ${Boost_container_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_container_LINK_LIBS_MINSIZEREL}
                ${Boost_container_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET Boost::container PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${Boost_container_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${Boost_container_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_container_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_container_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET Boost::container PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${Boost_container_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${Boost_container_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_container_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_container_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET Boost::container PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${Boost_container_COMPILE_OPTIONS_C_DEBUG}
                 ${Boost_container_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${Boost_container_COMPILE_OPTIONS_C_RELEASE}
                 ${Boost_container_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${Boost_container_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${Boost_container_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${Boost_container_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${Boost_container_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(Boost_container_TARGET_PROPERTIES TRUE)

########## COMPONENT chrono TARGET PROPERTIES ######################################

set_property(TARGET Boost::chrono PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${Boost_chrono_LINK_LIBS_DEBUG}
                ${Boost_chrono_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${Boost_chrono_LINK_LIBS_RELEASE}
                ${Boost_chrono_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_chrono_LINK_LIBS_RELWITHDEBINFO}
                ${Boost_chrono_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_chrono_LINK_LIBS_MINSIZEREL}
                ${Boost_chrono_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET Boost::chrono PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${Boost_chrono_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${Boost_chrono_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_chrono_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_chrono_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET Boost::chrono PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${Boost_chrono_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${Boost_chrono_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_chrono_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_chrono_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET Boost::chrono PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${Boost_chrono_COMPILE_OPTIONS_C_DEBUG}
                 ${Boost_chrono_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${Boost_chrono_COMPILE_OPTIONS_C_RELEASE}
                 ${Boost_chrono_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${Boost_chrono_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${Boost_chrono_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${Boost_chrono_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${Boost_chrono_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(Boost_chrono_TARGET_PROPERTIES TRUE)

########## COMPONENT atomic TARGET PROPERTIES ######################################

set_property(TARGET Boost::atomic PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${Boost_atomic_LINK_LIBS_DEBUG}
                ${Boost_atomic_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${Boost_atomic_LINK_LIBS_RELEASE}
                ${Boost_atomic_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_atomic_LINK_LIBS_RELWITHDEBINFO}
                ${Boost_atomic_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_atomic_LINK_LIBS_MINSIZEREL}
                ${Boost_atomic_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET Boost::atomic PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${Boost_atomic_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${Boost_atomic_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_atomic_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_atomic_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET Boost::atomic PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${Boost_atomic_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${Boost_atomic_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_atomic_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_atomic_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET Boost::atomic PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${Boost_atomic_COMPILE_OPTIONS_C_DEBUG}
                 ${Boost_atomic_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${Boost_atomic_COMPILE_OPTIONS_C_RELEASE}
                 ${Boost_atomic_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${Boost_atomic_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${Boost_atomic_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${Boost_atomic_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${Boost_atomic_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(Boost_atomic_TARGET_PROPERTIES TRUE)

########## COMPONENT thread TARGET PROPERTIES ######################################

set_property(TARGET Boost::thread PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${Boost_thread_LINK_LIBS_DEBUG}
                ${Boost_thread_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${Boost_thread_LINK_LIBS_RELEASE}
                ${Boost_thread_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_thread_LINK_LIBS_RELWITHDEBINFO}
                ${Boost_thread_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_thread_LINK_LIBS_MINSIZEREL}
                ${Boost_thread_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET Boost::thread PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${Boost_thread_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${Boost_thread_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_thread_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_thread_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET Boost::thread PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${Boost_thread_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${Boost_thread_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_thread_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_thread_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET Boost::thread PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${Boost_thread_COMPILE_OPTIONS_C_DEBUG}
                 ${Boost_thread_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${Boost_thread_COMPILE_OPTIONS_C_RELEASE}
                 ${Boost_thread_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${Boost_thread_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${Boost_thread_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${Boost_thread_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${Boost_thread_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(Boost_thread_TARGET_PROPERTIES TRUE)

########## COMPONENT type_erasure TARGET PROPERTIES ######################################

set_property(TARGET Boost::type_erasure PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${Boost_type_erasure_LINK_LIBS_DEBUG}
                ${Boost_type_erasure_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${Boost_type_erasure_LINK_LIBS_RELEASE}
                ${Boost_type_erasure_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_type_erasure_LINK_LIBS_RELWITHDEBINFO}
                ${Boost_type_erasure_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_type_erasure_LINK_LIBS_MINSIZEREL}
                ${Boost_type_erasure_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET Boost::type_erasure PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${Boost_type_erasure_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${Boost_type_erasure_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_type_erasure_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_type_erasure_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET Boost::type_erasure PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${Boost_type_erasure_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${Boost_type_erasure_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_type_erasure_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_type_erasure_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET Boost::type_erasure PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${Boost_type_erasure_COMPILE_OPTIONS_C_DEBUG}
                 ${Boost_type_erasure_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${Boost_type_erasure_COMPILE_OPTIONS_C_RELEASE}
                 ${Boost_type_erasure_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${Boost_type_erasure_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${Boost_type_erasure_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${Boost_type_erasure_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${Boost_type_erasure_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(Boost_type_erasure_TARGET_PROPERTIES TRUE)

########## COMPONENT timer TARGET PROPERTIES ######################################

set_property(TARGET Boost::timer PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${Boost_timer_LINK_LIBS_DEBUG}
                ${Boost_timer_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${Boost_timer_LINK_LIBS_RELEASE}
                ${Boost_timer_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_timer_LINK_LIBS_RELWITHDEBINFO}
                ${Boost_timer_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_timer_LINK_LIBS_MINSIZEREL}
                ${Boost_timer_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET Boost::timer PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${Boost_timer_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${Boost_timer_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_timer_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_timer_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET Boost::timer PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${Boost_timer_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${Boost_timer_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_timer_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_timer_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET Boost::timer PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${Boost_timer_COMPILE_OPTIONS_C_DEBUG}
                 ${Boost_timer_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${Boost_timer_COMPILE_OPTIONS_C_RELEASE}
                 ${Boost_timer_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${Boost_timer_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${Boost_timer_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${Boost_timer_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${Boost_timer_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(Boost_timer_TARGET_PROPERTIES TRUE)

########## COMPONENT stacktrace TARGET PROPERTIES ######################################

set_property(TARGET Boost::stacktrace PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${Boost_stacktrace_LINK_LIBS_DEBUG}
                ${Boost_stacktrace_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${Boost_stacktrace_LINK_LIBS_RELEASE}
                ${Boost_stacktrace_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_stacktrace_LINK_LIBS_RELWITHDEBINFO}
                ${Boost_stacktrace_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_stacktrace_LINK_LIBS_MINSIZEREL}
                ${Boost_stacktrace_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET Boost::stacktrace PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${Boost_stacktrace_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${Boost_stacktrace_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_stacktrace_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_stacktrace_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET Boost::stacktrace PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${Boost_stacktrace_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${Boost_stacktrace_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_stacktrace_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_stacktrace_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET Boost::stacktrace PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${Boost_stacktrace_COMPILE_OPTIONS_C_DEBUG}
                 ${Boost_stacktrace_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${Boost_stacktrace_COMPILE_OPTIONS_C_RELEASE}
                 ${Boost_stacktrace_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${Boost_stacktrace_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${Boost_stacktrace_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${Boost_stacktrace_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${Boost_stacktrace_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(Boost_stacktrace_TARGET_PROPERTIES TRUE)

########## COMPONENT stacktrace_noop TARGET PROPERTIES ######################################

set_property(TARGET Boost::stacktrace_noop PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${Boost_stacktrace_noop_LINK_LIBS_DEBUG}
                ${Boost_stacktrace_noop_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${Boost_stacktrace_noop_LINK_LIBS_RELEASE}
                ${Boost_stacktrace_noop_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_stacktrace_noop_LINK_LIBS_RELWITHDEBINFO}
                ${Boost_stacktrace_noop_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_stacktrace_noop_LINK_LIBS_MINSIZEREL}
                ${Boost_stacktrace_noop_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET Boost::stacktrace_noop PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${Boost_stacktrace_noop_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${Boost_stacktrace_noop_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_stacktrace_noop_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_stacktrace_noop_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET Boost::stacktrace_noop PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${Boost_stacktrace_noop_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${Boost_stacktrace_noop_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_stacktrace_noop_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_stacktrace_noop_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET Boost::stacktrace_noop PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${Boost_stacktrace_noop_COMPILE_OPTIONS_C_DEBUG}
                 ${Boost_stacktrace_noop_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${Boost_stacktrace_noop_COMPILE_OPTIONS_C_RELEASE}
                 ${Boost_stacktrace_noop_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${Boost_stacktrace_noop_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${Boost_stacktrace_noop_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${Boost_stacktrace_noop_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${Boost_stacktrace_noop_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(Boost_stacktrace_noop_TARGET_PROPERTIES TRUE)

########## COMPONENT stacktrace_basic TARGET PROPERTIES ######################################

set_property(TARGET Boost::stacktrace_basic PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${Boost_stacktrace_basic_LINK_LIBS_DEBUG}
                ${Boost_stacktrace_basic_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${Boost_stacktrace_basic_LINK_LIBS_RELEASE}
                ${Boost_stacktrace_basic_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_stacktrace_basic_LINK_LIBS_RELWITHDEBINFO}
                ${Boost_stacktrace_basic_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_stacktrace_basic_LINK_LIBS_MINSIZEREL}
                ${Boost_stacktrace_basic_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET Boost::stacktrace_basic PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${Boost_stacktrace_basic_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${Boost_stacktrace_basic_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_stacktrace_basic_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_stacktrace_basic_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET Boost::stacktrace_basic PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${Boost_stacktrace_basic_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${Boost_stacktrace_basic_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_stacktrace_basic_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_stacktrace_basic_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET Boost::stacktrace_basic PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${Boost_stacktrace_basic_COMPILE_OPTIONS_C_DEBUG}
                 ${Boost_stacktrace_basic_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${Boost_stacktrace_basic_COMPILE_OPTIONS_C_RELEASE}
                 ${Boost_stacktrace_basic_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${Boost_stacktrace_basic_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${Boost_stacktrace_basic_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${Boost_stacktrace_basic_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${Boost_stacktrace_basic_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(Boost_stacktrace_basic_TARGET_PROPERTIES TRUE)

########## COMPONENT stacktrace_backtrace TARGET PROPERTIES ######################################

set_property(TARGET Boost::stacktrace_backtrace PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${Boost_stacktrace_backtrace_LINK_LIBS_DEBUG}
                ${Boost_stacktrace_backtrace_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${Boost_stacktrace_backtrace_LINK_LIBS_RELEASE}
                ${Boost_stacktrace_backtrace_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_stacktrace_backtrace_LINK_LIBS_RELWITHDEBINFO}
                ${Boost_stacktrace_backtrace_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_stacktrace_backtrace_LINK_LIBS_MINSIZEREL}
                ${Boost_stacktrace_backtrace_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET Boost::stacktrace_backtrace PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${Boost_stacktrace_backtrace_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${Boost_stacktrace_backtrace_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_stacktrace_backtrace_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_stacktrace_backtrace_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET Boost::stacktrace_backtrace PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${Boost_stacktrace_backtrace_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${Boost_stacktrace_backtrace_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_stacktrace_backtrace_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_stacktrace_backtrace_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET Boost::stacktrace_backtrace PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${Boost_stacktrace_backtrace_COMPILE_OPTIONS_C_DEBUG}
                 ${Boost_stacktrace_backtrace_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${Boost_stacktrace_backtrace_COMPILE_OPTIONS_C_RELEASE}
                 ${Boost_stacktrace_backtrace_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${Boost_stacktrace_backtrace_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${Boost_stacktrace_backtrace_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${Boost_stacktrace_backtrace_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${Boost_stacktrace_backtrace_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(Boost_stacktrace_backtrace_TARGET_PROPERTIES TRUE)

########## COMPONENT stacktrace_addr2line TARGET PROPERTIES ######################################

set_property(TARGET Boost::stacktrace_addr2line PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${Boost_stacktrace_addr2line_LINK_LIBS_DEBUG}
                ${Boost_stacktrace_addr2line_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${Boost_stacktrace_addr2line_LINK_LIBS_RELEASE}
                ${Boost_stacktrace_addr2line_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_stacktrace_addr2line_LINK_LIBS_RELWITHDEBINFO}
                ${Boost_stacktrace_addr2line_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_stacktrace_addr2line_LINK_LIBS_MINSIZEREL}
                ${Boost_stacktrace_addr2line_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET Boost::stacktrace_addr2line PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${Boost_stacktrace_addr2line_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${Boost_stacktrace_addr2line_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_stacktrace_addr2line_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_stacktrace_addr2line_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET Boost::stacktrace_addr2line PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${Boost_stacktrace_addr2line_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${Boost_stacktrace_addr2line_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_stacktrace_addr2line_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_stacktrace_addr2line_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET Boost::stacktrace_addr2line PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${Boost_stacktrace_addr2line_COMPILE_OPTIONS_C_DEBUG}
                 ${Boost_stacktrace_addr2line_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${Boost_stacktrace_addr2line_COMPILE_OPTIONS_C_RELEASE}
                 ${Boost_stacktrace_addr2line_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${Boost_stacktrace_addr2line_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${Boost_stacktrace_addr2line_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${Boost_stacktrace_addr2line_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${Boost_stacktrace_addr2line_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(Boost_stacktrace_addr2line_TARGET_PROPERTIES TRUE)

########## COMPONENT regex TARGET PROPERTIES ######################################

set_property(TARGET Boost::regex PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${Boost_regex_LINK_LIBS_DEBUG}
                ${Boost_regex_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${Boost_regex_LINK_LIBS_RELEASE}
                ${Boost_regex_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_regex_LINK_LIBS_RELWITHDEBINFO}
                ${Boost_regex_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_regex_LINK_LIBS_MINSIZEREL}
                ${Boost_regex_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET Boost::regex PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${Boost_regex_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${Boost_regex_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_regex_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_regex_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET Boost::regex PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${Boost_regex_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${Boost_regex_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_regex_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_regex_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET Boost::regex PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${Boost_regex_COMPILE_OPTIONS_C_DEBUG}
                 ${Boost_regex_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${Boost_regex_COMPILE_OPTIONS_C_RELEASE}
                 ${Boost_regex_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${Boost_regex_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${Boost_regex_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${Boost_regex_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${Boost_regex_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(Boost_regex_TARGET_PROPERTIES TRUE)

########## COMPONENT random TARGET PROPERTIES ######################################

set_property(TARGET Boost::random PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${Boost_random_LINK_LIBS_DEBUG}
                ${Boost_random_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${Boost_random_LINK_LIBS_RELEASE}
                ${Boost_random_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_random_LINK_LIBS_RELWITHDEBINFO}
                ${Boost_random_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_random_LINK_LIBS_MINSIZEREL}
                ${Boost_random_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET Boost::random PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${Boost_random_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${Boost_random_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_random_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_random_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET Boost::random PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${Boost_random_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${Boost_random_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_random_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_random_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET Boost::random PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${Boost_random_COMPILE_OPTIONS_C_DEBUG}
                 ${Boost_random_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${Boost_random_COMPILE_OPTIONS_C_RELEASE}
                 ${Boost_random_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${Boost_random_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${Boost_random_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${Boost_random_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${Boost_random_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(Boost_random_TARGET_PROPERTIES TRUE)

########## COMPONENT program_options TARGET PROPERTIES ######################################

set_property(TARGET Boost::program_options PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${Boost_program_options_LINK_LIBS_DEBUG}
                ${Boost_program_options_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${Boost_program_options_LINK_LIBS_RELEASE}
                ${Boost_program_options_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_program_options_LINK_LIBS_RELWITHDEBINFO}
                ${Boost_program_options_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_program_options_LINK_LIBS_MINSIZEREL}
                ${Boost_program_options_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET Boost::program_options PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${Boost_program_options_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${Boost_program_options_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_program_options_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_program_options_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET Boost::program_options PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${Boost_program_options_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${Boost_program_options_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_program_options_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_program_options_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET Boost::program_options PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${Boost_program_options_COMPILE_OPTIONS_C_DEBUG}
                 ${Boost_program_options_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${Boost_program_options_COMPILE_OPTIONS_C_RELEASE}
                 ${Boost_program_options_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${Boost_program_options_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${Boost_program_options_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${Boost_program_options_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${Boost_program_options_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(Boost_program_options_TARGET_PROPERTIES TRUE)

########## COMPONENT filesystem TARGET PROPERTIES ######################################

set_property(TARGET Boost::filesystem PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${Boost_filesystem_LINK_LIBS_DEBUG}
                ${Boost_filesystem_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${Boost_filesystem_LINK_LIBS_RELEASE}
                ${Boost_filesystem_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_filesystem_LINK_LIBS_RELWITHDEBINFO}
                ${Boost_filesystem_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_filesystem_LINK_LIBS_MINSIZEREL}
                ${Boost_filesystem_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET Boost::filesystem PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${Boost_filesystem_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${Boost_filesystem_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_filesystem_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_filesystem_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET Boost::filesystem PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${Boost_filesystem_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${Boost_filesystem_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_filesystem_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_filesystem_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET Boost::filesystem PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${Boost_filesystem_COMPILE_OPTIONS_C_DEBUG}
                 ${Boost_filesystem_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${Boost_filesystem_COMPILE_OPTIONS_C_RELEASE}
                 ${Boost_filesystem_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${Boost_filesystem_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${Boost_filesystem_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${Boost_filesystem_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${Boost_filesystem_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(Boost_filesystem_TARGET_PROPERTIES TRUE)

########## COMPONENT log TARGET PROPERTIES ######################################

set_property(TARGET Boost::log PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${Boost_log_LINK_LIBS_DEBUG}
                ${Boost_log_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${Boost_log_LINK_LIBS_RELEASE}
                ${Boost_log_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_log_LINK_LIBS_RELWITHDEBINFO}
                ${Boost_log_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_log_LINK_LIBS_MINSIZEREL}
                ${Boost_log_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET Boost::log PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${Boost_log_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${Boost_log_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_log_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_log_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET Boost::log PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${Boost_log_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${Boost_log_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_log_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_log_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET Boost::log PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${Boost_log_COMPILE_OPTIONS_C_DEBUG}
                 ${Boost_log_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${Boost_log_COMPILE_OPTIONS_C_RELEASE}
                 ${Boost_log_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${Boost_log_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${Boost_log_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${Boost_log_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${Boost_log_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(Boost_log_TARGET_PROPERTIES TRUE)

########## COMPONENT log_setup TARGET PROPERTIES ######################################

set_property(TARGET Boost::log_setup PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${Boost_log_setup_LINK_LIBS_DEBUG}
                ${Boost_log_setup_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${Boost_log_setup_LINK_LIBS_RELEASE}
                ${Boost_log_setup_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_log_setup_LINK_LIBS_RELWITHDEBINFO}
                ${Boost_log_setup_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_log_setup_LINK_LIBS_MINSIZEREL}
                ${Boost_log_setup_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET Boost::log_setup PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${Boost_log_setup_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${Boost_log_setup_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_log_setup_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_log_setup_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET Boost::log_setup PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${Boost_log_setup_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${Boost_log_setup_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_log_setup_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_log_setup_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET Boost::log_setup PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${Boost_log_setup_COMPILE_OPTIONS_C_DEBUG}
                 ${Boost_log_setup_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${Boost_log_setup_COMPILE_OPTIONS_C_RELEASE}
                 ${Boost_log_setup_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${Boost_log_setup_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${Boost_log_setup_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${Boost_log_setup_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${Boost_log_setup_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(Boost_log_setup_TARGET_PROPERTIES TRUE)

########## COMPONENT iostreams TARGET PROPERTIES ######################################

set_property(TARGET Boost::iostreams PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${Boost_iostreams_LINK_LIBS_DEBUG}
                ${Boost_iostreams_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${Boost_iostreams_LINK_LIBS_RELEASE}
                ${Boost_iostreams_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_iostreams_LINK_LIBS_RELWITHDEBINFO}
                ${Boost_iostreams_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_iostreams_LINK_LIBS_MINSIZEREL}
                ${Boost_iostreams_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET Boost::iostreams PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${Boost_iostreams_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${Boost_iostreams_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_iostreams_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_iostreams_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET Boost::iostreams PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${Boost_iostreams_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${Boost_iostreams_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_iostreams_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_iostreams_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET Boost::iostreams PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${Boost_iostreams_COMPILE_OPTIONS_C_DEBUG}
                 ${Boost_iostreams_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${Boost_iostreams_COMPILE_OPTIONS_C_RELEASE}
                 ${Boost_iostreams_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${Boost_iostreams_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${Boost_iostreams_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${Boost_iostreams_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${Boost_iostreams_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(Boost_iostreams_TARGET_PROPERTIES TRUE)

########## COMPONENT context TARGET PROPERTIES ######################################

set_property(TARGET Boost::context PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${Boost_context_LINK_LIBS_DEBUG}
                ${Boost_context_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${Boost_context_LINK_LIBS_RELEASE}
                ${Boost_context_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_context_LINK_LIBS_RELWITHDEBINFO}
                ${Boost_context_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_context_LINK_LIBS_MINSIZEREL}
                ${Boost_context_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET Boost::context PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${Boost_context_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${Boost_context_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_context_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_context_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET Boost::context PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${Boost_context_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${Boost_context_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_context_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_context_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET Boost::context PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${Boost_context_COMPILE_OPTIONS_C_DEBUG}
                 ${Boost_context_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${Boost_context_COMPILE_OPTIONS_C_RELEASE}
                 ${Boost_context_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${Boost_context_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${Boost_context_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${Boost_context_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${Boost_context_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(Boost_context_TARGET_PROPERTIES TRUE)

########## COMPONENT coroutine TARGET PROPERTIES ######################################

set_property(TARGET Boost::coroutine PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${Boost_coroutine_LINK_LIBS_DEBUG}
                ${Boost_coroutine_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${Boost_coroutine_LINK_LIBS_RELEASE}
                ${Boost_coroutine_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_coroutine_LINK_LIBS_RELWITHDEBINFO}
                ${Boost_coroutine_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_coroutine_LINK_LIBS_MINSIZEREL}
                ${Boost_coroutine_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET Boost::coroutine PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${Boost_coroutine_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${Boost_coroutine_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_coroutine_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_coroutine_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET Boost::coroutine PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${Boost_coroutine_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${Boost_coroutine_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_coroutine_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_coroutine_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET Boost::coroutine PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${Boost_coroutine_COMPILE_OPTIONS_C_DEBUG}
                 ${Boost_coroutine_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${Boost_coroutine_COMPILE_OPTIONS_C_RELEASE}
                 ${Boost_coroutine_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${Boost_coroutine_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${Boost_coroutine_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${Boost_coroutine_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${Boost_coroutine_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(Boost_coroutine_TARGET_PROPERTIES TRUE)

########## COMPONENT contract TARGET PROPERTIES ######################################

set_property(TARGET Boost::contract PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${Boost_contract_LINK_LIBS_DEBUG}
                ${Boost_contract_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${Boost_contract_LINK_LIBS_RELEASE}
                ${Boost_contract_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_contract_LINK_LIBS_RELWITHDEBINFO}
                ${Boost_contract_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_contract_LINK_LIBS_MINSIZEREL}
                ${Boost_contract_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET Boost::contract PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${Boost_contract_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${Boost_contract_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_contract_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_contract_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET Boost::contract PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${Boost_contract_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${Boost_contract_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_contract_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_contract_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET Boost::contract PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${Boost_contract_COMPILE_OPTIONS_C_DEBUG}
                 ${Boost_contract_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${Boost_contract_COMPILE_OPTIONS_C_RELEASE}
                 ${Boost_contract_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${Boost_contract_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${Boost_contract_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${Boost_contract_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${Boost_contract_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(Boost_contract_TARGET_PROPERTIES TRUE)

########## COMPONENT boost TARGET PROPERTIES ######################################

set_property(TARGET Boost::boost PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Debug>:${Boost_boost_LINK_LIBS_DEBUG}
                ${Boost_boost_LINKER_FLAGS_LIST_DEBUG}>
             $<$<CONFIG:Release>:${Boost_boost_LINK_LIBS_RELEASE}
                ${Boost_boost_LINKER_FLAGS_LIST_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_boost_LINK_LIBS_RELWITHDEBINFO}
                ${Boost_boost_LINKER_FLAGS_LIST_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_boost_LINK_LIBS_MINSIZEREL}
                ${Boost_boost_LINKER_FLAGS_LIST_MINSIZEREL}>)
set_property(TARGET Boost::boost PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             $<$<CONFIG:Debug>:${Boost_boost_INCLUDE_DIRS_DEBUG}>
             $<$<CONFIG:Release>:${Boost_boost_INCLUDE_DIRS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_boost_INCLUDE_DIRS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_boost_INCLUDE_DIRS_MINSIZEREL}>)
set_property(TARGET Boost::boost PROPERTY INTERFACE_COMPILE_DEFINITIONS
             $<$<CONFIG:Debug>:${Boost_boost_COMPILE_DEFINITIONS_DEBUG}>
             $<$<CONFIG:Release>:${Boost_boost_COMPILE_DEFINITIONS_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:${Boost_boost_COMPILE_DEFINITIONS_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:${Boost_boost_COMPILE_DEFINITIONS_MINSIZEREL}>)
set_property(TARGET Boost::boost PROPERTY INTERFACE_COMPILE_OPTIONS
             $<$<CONFIG:Debug>:
                 ${Boost_boost_COMPILE_OPTIONS_C_DEBUG}
                 ${Boost_boost_COMPILE_OPTIONS_CXX_DEBUG}>
             $<$<CONFIG:Release>:
                 ${Boost_boost_COMPILE_OPTIONS_C_RELEASE}
                 ${Boost_boost_COMPILE_OPTIONS_CXX_RELEASE}>
             $<$<CONFIG:RelWithDebInfo>:
                 ${Boost_boost_COMPILE_OPTIONS_C_RELWITHDEBINFO}
                 ${Boost_boost_COMPILE_OPTIONS_CXX_RELWITHDEBINFO}>
             $<$<CONFIG:MinSizeRel>:
                 ${Boost_boost_COMPILE_OPTIONS_C_MINSIZEREL}
                 ${Boost_boost_COMPILE_OPTIONS_CXX_MINSIZEREL}>)
set(Boost_boost_TARGET_PROPERTIES TRUE)

########## GLOBAL TARGET PROPERTIES #########################################################

if(NOT Boost_Boost_TARGET_PROPERTIES)
    set_property(TARGET Boost::Boost APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                 $<$<CONFIG:Debug>:${Boost_COMPONENTS_DEBUG}>
                 $<$<CONFIG:Release>:${Boost_COMPONENTS_RELEASE}>
                 $<$<CONFIG:RelWithDebInfo>:${Boost_COMPONENTS_RELWITHDEBINFO}>
                 $<$<CONFIG:MinSizeRel>:${Boost_COMPONENTS_MINSIZEREL}>)
endif()

########## BUILD MODULES ####################################################################
#############################################################################################

########## COMPONENT dynamic_linking BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${Boost_dynamic_linking_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_dynamic_linking_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_dynamic_linking_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_dynamic_linking_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT disable_autolinking BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${Boost_disable_autolinking_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_disable_autolinking_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_disable_autolinking_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_disable_autolinking_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT diagnostic_definitions BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${Boost_diagnostic_definitions_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_diagnostic_definitions_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_diagnostic_definitions_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_diagnostic_definitions_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT headers BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${Boost_headers_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_headers_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_headers_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_headers_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT _libboost BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${Boost__libboost_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost__libboost_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost__libboost_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost__libboost_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT serialization BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${Boost_serialization_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_serialization_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_serialization_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_serialization_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT wserialization BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${Boost_wserialization_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_wserialization_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_wserialization_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_wserialization_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT exception BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${Boost_exception_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_exception_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_exception_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_exception_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT test BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${Boost_test_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_test_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_test_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_test_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT test_exec_monitor BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${Boost_test_exec_monitor_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_test_exec_monitor_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_test_exec_monitor_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_test_exec_monitor_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT prg_exec_monitor BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${Boost_prg_exec_monitor_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_prg_exec_monitor_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_prg_exec_monitor_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_prg_exec_monitor_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT unit_test_framework BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${Boost_unit_test_framework_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_unit_test_framework_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_unit_test_framework_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_unit_test_framework_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT system BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${Boost_system_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_system_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_system_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_system_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT date_time BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${Boost_date_time_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_date_time_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_date_time_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_date_time_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT container BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${Boost_container_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_container_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_container_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_container_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT chrono BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${Boost_chrono_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_chrono_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_chrono_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_chrono_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT atomic BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${Boost_atomic_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_atomic_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_atomic_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_atomic_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT thread BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${Boost_thread_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_thread_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_thread_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_thread_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT type_erasure BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${Boost_type_erasure_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_type_erasure_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_type_erasure_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_type_erasure_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT timer BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${Boost_timer_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_timer_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_timer_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_timer_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT stacktrace BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${Boost_stacktrace_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_stacktrace_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_stacktrace_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_stacktrace_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT stacktrace_noop BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${Boost_stacktrace_noop_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_stacktrace_noop_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_stacktrace_noop_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_stacktrace_noop_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT stacktrace_basic BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${Boost_stacktrace_basic_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_stacktrace_basic_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_stacktrace_basic_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_stacktrace_basic_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT stacktrace_backtrace BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${Boost_stacktrace_backtrace_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_stacktrace_backtrace_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_stacktrace_backtrace_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_stacktrace_backtrace_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT stacktrace_addr2line BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${Boost_stacktrace_addr2line_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_stacktrace_addr2line_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_stacktrace_addr2line_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_stacktrace_addr2line_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT regex BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${Boost_regex_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_regex_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_regex_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_regex_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT random BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${Boost_random_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_random_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_random_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_random_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT program_options BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${Boost_program_options_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_program_options_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_program_options_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_program_options_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT filesystem BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${Boost_filesystem_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_filesystem_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_filesystem_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_filesystem_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT log BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${Boost_log_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_log_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_log_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_log_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT log_setup BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${Boost_log_setup_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_log_setup_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_log_setup_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_log_setup_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT iostreams BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${Boost_iostreams_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_iostreams_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_iostreams_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_iostreams_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT context BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${Boost_context_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_context_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_context_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_context_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT coroutine BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${Boost_coroutine_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_coroutine_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_coroutine_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_coroutine_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT contract BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${Boost_contract_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_contract_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_contract_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_contract_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()

########## COMPONENT boost BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${Boost_boost_BUILD_MODULES_PATHS_DEBUG})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_boost_BUILD_MODULES_PATHS_RELEASE})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_boost_BUILD_MODULES_PATHS_RELWITHDEBINFO})
    include(${_BUILD_MODULE_PATH})
endforeach()

foreach(_BUILD_MODULE_PATH ${Boost_boost_BUILD_MODULES_PATHS_MINSIZEREL})
    include(${_BUILD_MODULE_PATH})
endforeach()