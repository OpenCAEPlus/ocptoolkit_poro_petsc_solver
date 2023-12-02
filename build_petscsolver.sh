#!/bin/bash
source /sh3/software/compiler/intel/2022/setvars.sh --force
module load gcc/7.3.0
module load cmake/3.23.1


export CPATH=/sh3/home/sh3sce0588/lsz/petsc/include:/sh3/home/sh3sce0588/lsz/petsc/int64_hypre_release/include:/sh3/home/sh3sce0588/lsz/lapack-3.11/CBLAS/include:/sh3/home/sh3sce0588/lsz/lapack-3.11/LAPACKE/include:$CPATH
export LD_LIBRARY_PATH=/sh3/home/sh3sce0588/lsz/lapack-3.11:$LD_LIBRARY_PATH

rm -rf build 
mkdir build
cd build
cmake ..
make
mv libpetsc_solver.a ../lib/


