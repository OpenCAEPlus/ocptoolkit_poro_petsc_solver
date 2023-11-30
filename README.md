# OpenCAX+ Toolkit Template

[![Toolkit Build](https://github.com/OpenCAXPlus/ocptk/actions/workflows/build.yml/badge.svg)](https://github.com/OpenCAXPlus/ocptk/actions/workflows/build.yml)
[![Toolkit Publish](https://github.com/OpenCAXPlus/ocptk/actions/workflows/publish.yml/badge.svg)](https://github.com/OpenCAXPlus/ocptk/actions/workflows/publish.yml)

This is a template repository for OpenCAX+ toolkit, which is a plugin that connects any external source codes to the OpenCAX+ SDK and take the advantage of the whole OpenCAX+ ecosystem. 

When the SDK use a toolkit, it will only know about the toolkit itself. It is the toolkit developers responsibility to create its dependencies on external projects. OCP only provides an online location for you to store the compressed file of external source codes.

## Learn how to create toolkit for OpenCAX+
Read our tutorial or watch the tutorial video (both coming soon).

## Things that you need to modify
- [ ] This README.md file, also remember to add the workflow badge
- [ ] The workflow yml files, change the name of workflow
- [ ] The ocp.yml file
- [ ] scripts/publish.sh
- [ ] scripts/prepare.sh
- [ ] Add source codes to the src folder
- [ ] Add necessary presets to CMakePresets.json file
- [ ] Add necessary config scripts

## Things you need to check before publish the toolkit project
- [ ] run the prepare.sh script locally, is the external project's source code resides in ocp/toolkit/toolkit_id/version/source?
- [ ] does the ocp/toolkit/toolkit_id/version folder contains a ocp.yml file and any build configuration folders that you want?
- [ ] run the ocp/toolkit/toolkit_id/version/configuration/install.sh script locally, is the build cache files put into ocp/external/external_id/version/configuration/build folder, and the install files put into ocp/external/external_id/version/configuration/install folder?
- [ ] run the Toolkit Build workflow
- [ ] must use CMake as your build system


```sh
cmake --preset="linux-gnu-Debug" -S "."
cmake --build --preset="linux-gnu-Debug" --target package_source
```