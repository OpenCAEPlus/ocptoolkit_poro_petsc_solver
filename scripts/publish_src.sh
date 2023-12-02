#!/bin/bash

version=${1:-"0.1.0"}
toolkit_id=${2:-"poro_petsc_solver"}

# current_date=$(date +"%Y.%m%d")
# short_hash=$(git rev-parse --short HEAD)
# dir="ocp/toolkit/$toolkit_id/$current_date.$short_hash/source"
# xzfile="${toolkit_id}-$current_date.$short_hash.tar.xz"

dir="ocp/toolkit/$toolkit_id/$version/source"
xzfile="${toolkit_id}-$version.tar.xz"


# move to the root of the external repository
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd $script_dir/..

# start the script
rm -f $xzfile
tar -cJf $xzfile ocp/
echo "Create compressed toolkit from ocp folder"
rclone copy -P $xzfile ali:ocp-toolkit/$toolkit_id
echo "Upload everything"

# move back to the folder where you started
cd -
