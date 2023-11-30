set(SOFTWARE spdlog)
set(VERSION 1.11.0)
set(ROOT $ENV{HOME}/ocp/external)

set(spdlog_DIR ${ROOT}/${SOFTWARE}/${VERSION}/install/default/lib/cmake/spdlog)
find_package(spdlog)

set(tkname ocptk)
add_library(${tkname} STATIC)
add_library(OCP::${tkname} ALIAS ${tkname})

target_sources(
  ${tkname}
  PRIVATE tk.cpp
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/tk.h>
         $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/tk.h>)
set_target_properties(${tkname} PROPERTIES PUBLIC_HEADER tk.h)

target_link_libraries(${tkname} PUBLIC spdlog::spdlog) # OCP::Interface)
target_include_directories(
  ${tkname} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                   $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

include(GNUInstallDirs)

install(
  TARGETS ${tkname}
  EXPORT ${tkname}Targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  INCLUDES
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${tkname}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${tkname})

install(
  EXPORT ${tkname}Targets
  FILE ${tkname}Targets.cmake
  NAMESPACE ocp::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ocptk)

export(
  EXPORT ${tkname}Targets
  FILE "${CMAKE_CURRENT_BINARY_DIR}/${tkname}Targets.cmake"
  NAMESPACE ocp::)
