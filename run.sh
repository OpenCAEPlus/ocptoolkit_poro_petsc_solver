#!bin/sh

# source ~/.bashrc
# source /opt/intel/bin/compilervars.sh intel64
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/matlab/bin/glnxa64:/opt/matlab/sys/os/glnxa64


# ############ argparser  ###########
# until [ $# -eq 0 ]
# do
#   name=${1:1}; shift;
#   if [[ -z "$1" || $1 == -* ]] ; then eval "export $name=true"; else eval "export $name=$1"; shift; fi
# done
# ########## set default  ###########
# if [ ${#nt} == 0 ]
# 	then
# 	    nt=1;  # default value to give    
# else
# 	    nt=$nt;
# fi

# # create log
# if [ ! -d "./log" ];then
# 	mkdir log 
# fi
# # create TA
# if [ ! -d "./TA" ];then
# 	mkdir TA 
# fi

# # export OMP_NUM_THREADS=8
# sh clean.sh
# make clean
# make -j 4
# # sh clean.sh

# # valgrind --tool=memcheck --leak-check=full
# # valgrind --leak-check=full 

# # ./TriAngels workpath="../算例/峭面磁扩散波/h0.005" input="计算输入文件.txt"  #2>&1 |tee -a ./log/run-ex1.log
# # ./TriAngels workpath="../算例/峭面磁扩散波/40x40" input="计算输入文件.txt"  #2>&1 |tee -a ./log/run-ex1.log
# # ./TriAngels workpath="../算例/峭面磁扩散波/80x80" input="计算输入文件.txt"  #2>&1 |tee -a ./log/run-ex1.log


# # ./TriAngels workpath="../算例/峭面磁扩散波/10x10" input="计算输入文件.txt"  #2>&1 |tee -a ./log/run-ex1.log
# # ./TriAngels workpath="../算例/MagData/10x10" input="计算输入文件.txt" # 2>&1 |tee ./log/run-10x10.log
# # ./TriAngels workpath="../算例/MagData/20x20" input="计算输入文件.txt"  #2>&1 |tee ./log/run-20x20.log
# # ./TriAngels workpath="../算例/MagData/40x40" input="计算输入文件.txt"  #2>&1 |tee ./log/run-40x40-ar-init-noavg-am10.log
# # ./TriAngels workpath="../算例/MagData/80x80" input="计算输入文件.txt"  #2>&1 |tee run-80x80-dt1E-5-T05-F2C-20221206.log
# # ./TriAngels workpath="../算例/MagData/160x160" input="计算输入文件.txt"  2>&1 |tee ./log/run-160x160.log
# # ./TriAngels workpath="../算例/MagData/320x320" input="计算输入文件.txt"  #2>&1 |tee ./log/run-320x320.log
# # ./TriAngels workpath="../算例/MagData/640x640" input="计算输入文件.txt"  2>&1 |tee ./log/run-640x640.log
# # ./TriAngels workpath="../算例/MagData/1280x1280" input="计算输入文件.txt"  #2>&1 |tee ./log/run-ex1.log
# # ./TriAngels workpath="../算例/MagData/160x160-5" input="计算输入文件.txt"  2>&1 |tee ./log/run-ex1.log


# # ./TriAngels workpath="../算例/MagData/gmsh-h-0.125-80-Front" input="计算输入文件.txt"  #2>&1 |tee ./log/run-mesh-h0d00625-IMP.log

# ./TriAngels workpath="../算例/MagData/ansys-h-0.00625" input="计算输入文件.txt"  2>&1 |tee ./log/run-mesh-h0d00625-refine6-F2F.log
# # ./TriAngels workpath="../算例/MagData/gmsh-h-0.00625" input="计算输入文件.txt"  #2>&1 |tee ./log/run-mesh-h0d00625-IMP.log

# # ./TriAngels workpath="../算例/MagData/80x80-1" input="计算输入文件.txt"  2>&1 |tee ./log/run-mesh-80x80-1-FIM.log
# # ./TriAngels workpath="../算例/MagData/79x81-5-down" input="计算输入文件.txt"  #2>&1 |tee ./log/run-mesh-80x80-1.log

# # % theta =  5°, Nx = 23 , Ny = 271
# # % theta = 15°, Nx = 42 , Ny = 156
# # % theta = 30°, Nx = 61 , Ny = 105
# # % theta = 45°, Nx = 80 , Ny = 80
# # % theta = 60°, Nx = 105, Ny = 61
# # % theta = 75°, Nx = 156, Ny = 42
# # % theta = 85°, Nx = 271, Ny = 23
# # ./TriAngels workpath="../算例/MagData/23x271-1" input="计算输入文件.txt"  #2>&1 |tee ./log/run-mesh-80x80-1.log
# # ./TriAngels workpath="../算例/MagData/42x156-1" input="计算输入文件.txt"  #2>&1 |tee ./log/run-mesh-80x80-1.log
# # ./TriAngels workpath="../算例/MagData/61x105-1" input="计算输入文件.txt"  #2>&1 |tee ./log/run-mesh-80x80-1.log
# # ./TriAngels workpath="../算例/MagData/105x61-1" input="计算输入文件.txt"  #2>&1 |tee ./log/run-mesh-80x80-1.log
# # ./TriAngels workpath="../算例/MagData/156x42-1" input="计算输入文件.txt"  #2>&1 |tee ./log/run-mesh-80x80-1.log
# # ./TriAngels workpath="../算例/MagData/271x23-1" input="计算输入文件.txt"  #2>&1 |tee ./log/run-mesh-80x80-1.log

make -j 4

# mv /home/spring/zlj/simfast_mpi/ParPennSim-4-14-16/lib/libparpennsim.a /home/spring/zlj/simfast_mpi/petsc-3.6.3/src/ksp/ksp/examples/tests/
mv /home/spring/zlj/simfast_mpi/ParPennSim-4-14-16/lib/libparpennsim.a /home/spring/zlj/simfast_mpi/petsc-3.6.3/src/ksp/ksp/examples/tutorials/
