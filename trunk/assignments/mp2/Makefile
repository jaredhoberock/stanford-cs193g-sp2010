# A simple CUDA makefile.
# USAGE:
#  compile:
#	make all	// compiles all the parts
#
#  run:
#	make run1	// runs part 1
#	make run2	// runs part 2
#	make run3	// runs part 3

# CUDA depends on two things:
#  1) The CUDA nvcc compiler, which needs to be on your path,
#	or called directly, which we do here
#  2) The CUDA shared library being available at runtime,
#	which we make available by setting the LD_LIBRARY_PATH
#	variable for the durection of the makefile.
#
# You can set your PATH and LD_LIBRARY_PATH variables as part of your
# .profile so that you can compile and run without using this makefile.

NVCCFLAGS	:= -O3 -Xptxas -v 
NVCC		:= /usr/local/cuda/bin/nvcc
LD_LIBRARY_PATH	:= /usr/local/cuda/lib64


all: part1 part2 part3

part1: mp2-part1.cu mp2-util.h
	$(NVCC) -arch=sm_11 -o part1 mp2-part1.cu $(NVCCFLAGS)

part2: mp2-part2.cu mp2-util.h
	$(NVCC) -o part2 mp2-part2.cu $(NVCCFLAGS)
	
part3: mp2-part3.cu mp2-part3-reference.cu mp2-util.h mp2.h
	$(NVCC) -arch=sm_11 -o part3 mp2-part3.cu mp2-part3-reference.cu $(NVCCFLAGS)
	
clean:
	rm -rf part1 part2 part3
