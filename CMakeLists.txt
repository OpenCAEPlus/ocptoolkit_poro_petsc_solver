cmake_minimum_required(VERSION 3.1)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default configuration types and flags
set(CMAKE_CONFIGURATION_TYPES "Release;Debug" CACHE STRING "Configs" FORCE)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -g")

set(build_type "Release")
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "${build_type}" CACHE STRING "the build type" FORCE)
endif()
message(STATUS "Set build type to ${CMAKE_BUILD_TYPE} configuration")

# set(PETSC_DIR "/sh3/home/sh3sce0588/lsz/petsc")
# set(PETSC_DIR "/sh3/home/sh3sce0588/lsz/petsc/") #lsz
# set(PETSC_ARCH "int64_hypre_release") #lsz

set(PETSC_DIR "/sh3/home/sh3sce0588/lsz/petsc/" CACHE PATH "Path to PETSC")
set(PETSC_ARCH "int64_hypre_release" CACHE STRING "The PETSC architecture") #lsz

project(poro_petsc_solver VERSION 0.1.0)
# include(CTest) # add ctest support
# include(Version) # set the versioning convention is YYYY.mmdd.commit_short_hash
include(CPackConfig) # add cpack configuration from cmake folder

include_directories(${PROJECT_SOURCE_DIR}/src/)
include_directories(${PROJECT_SOURCE_DIR}/include/)
# include_directories(${PETSC_DIR}/include/)
# include_directories(${PETSC_DIR}/${PETSC_ARCH}/include/) #zhaoli, no modify
# include_directories(/public1/home/sch10084/ParPennSim-4-14-16/common/include/)
# include_directories(/public1/home/sch10084/ParPennSim-4-14-16/common/src/)
# include_directories(/public1/home/sch10084/ParPennSim-4-14-16/Utility/include/)
# include_directories(/public1/home/sch10084/ParPennSim-4-14-16/Utility/src/)
# include_directories(/home/xcheng/code/OpenCAXPlus/ocpexternal_lapack/ocp/external/lapack/3.11.0/install/intel/int64/include)
file(GLOB_RECURSE SRC_FILES ${PROJECT_SOURCE_DIR}/src/*.cpp)
set(tkname poro_petsc_solver)

add_library(${tkname} STATIC ${SRC_FILES})


find_package(PkgConfig)
pkg_check_modules(PETSC REQUIRED IMPORTED_TARGET PETSc)
pkg_check_modules(CBLAS REQUIRED IMPORTED_TARGET cblas)



# Check for header file
# find_path(PETSC_INCLUDE_DIRS petscksp.h
#    HINTS ${PETSC_DIR}/include $ENV{PETSC_DIR}/include ${PROJECT_SOURCE_DIR}/petsc/include
#    DOC "Directory where the PETSC header is located")
# mark_as_advanced(PETSC_INCLUDE_DIRS)

# # Check for petsc library
# find_library(PETSC_LIBRARIES petsc
#     HINTS ${PETSC_DIR}/${PETSC_ARCH}/lib $ENV{PETSC_DIR}/lib ${PROJECT_SOURCE_DIR}/PETSC/lib
#     DOC "The PETSC library")
# mark_as_advanced(PETSC_LIBRARIES)

# set(PETSC_LIBRARIES ${PETSC_LIBRARIES})
# set(PETSC_INCLUDE_DIRS ${PETSC_INCLUDE_DIRS})

# add_library(petsc_solver STATIC IMPORTED GLOBAL)
# set_property(
#     TARGET petsc_solver 
#     PROPERTY IMPORTED_LOCATION ${PETSC_LIBRARIES})
#   set_property(
#     TARGET petsc_solver 
#     PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${PETSC_INCLUDE_DIRS})

message(STATUS ${PETSC_INCLUDE_DIRS})
message(STATUS ${PETSC_LINK_LIBRARIES})
message(STATUS ${PETSC_DIR})

target_link_libraries(${tkname} PRIVATE PkgConfig::PETSC PkgConfig::CBLAS)
# target_include_directories(petsc_solver PUBLIC ${PETSC_INCLUDE_DIRS})
target_compile_definitions(${tkname} PRIVATE PSINT64)
include(GNUInstallDirs)

# Standard package handling
include(FindPackageHandleStandardArgs)
# find_package_handle_standard_args(petsc_solver
#     "PETSC could not be found. Check PETSC_DIR."
#     PETSC_LIBRARIES PETSC_INCLUDE_DIRS)


###############################################################################
## Installtion targets for lib and executable files
###############################################################################

# Set install location for petsc_solver library
# set(CMAKE_INSTALL_LIBDIR "${PROJECT_SOURCE_DIR}/lib/")
# install(TARGETS petsc_solver
#         LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
#         ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

 
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
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${tkname})

export(
  EXPORT ${tkname}Targets
  FILE "${CMAKE_CURRENT_BINARY_DIR}/${tkname}Targets.cmake"
  NAMESPACE ocp::)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  ${PROJECT_NAME}ConfigVersion.cmake
  VERSION ${PACKAGE_VERSION}
  COMPATIBILITY AnyNewerVersion)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/${tkname}Config.cmake.in
               ${tkname}Config.cmake @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${tkname}Config.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/${tkname}ConfigVersion.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})
