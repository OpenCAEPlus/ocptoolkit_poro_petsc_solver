#!/bin/bash

version=${1:-"0.1.0"}
toolkit_id=${2:-"poro_petsc_solver"}

# current_date=$(date +"%Y.%m%d")
# short_hash=$(git rev-parse --short HEAD)
# dir="ocp/toolkit/$toolkit_id/$current_date.$short_hash/source"
# xzfile="${toolkit_id}-$current_date.$short_hash.tar.xz"

dir="ocp/toolkit/$toolkit_id/$version/source"
xzfile="${toolkit_id}-$version.tar.xz"


script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd $script_dir/..

rm -rf ocp
rm -rf $toolkit_id
rm -rf ${toolkit_id}-*.tar.xz

mkdir -p "$dir"

# cp -r src $dir
# cp ocp.toml $dir
# cp CMakeLists.txt $dir
# cp CMakePresets.json $dir

cmake --preset="linux-intel-Release" -S "."
cmake --build --preset="linux-intel-Release" --target package_source
tar -xJf build/linux-intel-Release/$xzfile -C $dir --strip-components=1
mv $dir/ocp.yml $dir/..
mv $dir/configurations $dir/..

# tar -cJf $xzfile ocp/

# mkdir -p $toolkit_id
# cp $xzfile $toolkit_id
# cp ocp.toml $toolkit_id

cd -
