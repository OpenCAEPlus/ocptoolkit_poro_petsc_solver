cmake_minimum_required(VERSION 3.20)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

project(
  ocptk
  DESCRIPTION "The OpenCAX+ toolkit template"
  LANGUAGES C CXX Fortran)

include(CTest) # add ctest support
include(Version) # set the versioning convention is YYYY.mmdd.commit_short_hash
include(CPackConfig) # add cpack configuration from cmake folder
# include(CPackSourceConfig)

# option(DEPENDS_EXTERNAL "If the toolkit depends on external projects" OFF)

# if(DEPENDS_EXTERNAL) # prepare the depended external project if necessary #
# Pull external project if necessary Use the ocp cli to download from url Use #
# the ocp cli to install external project endif()

# add_subdirectory(external) # remove if toolkit does not depend on external
# projects

# set(OCP_DIR $ENV{HOME}/ocp/sdk) find_package(OCP)

add_subdirectory(src)

include(GNUInstallDirs)

install(FILES "LICENSE"
        DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/${PROJECT_NAME}")

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  ${PROJECT_NAME}ConfigVersion.cmake
  VERSION ${PACKAGE_VERSION}
  COMPATIBILITY AnyNewerVersion)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in
               ${PROJECT_NAME}Config.cmake @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})
