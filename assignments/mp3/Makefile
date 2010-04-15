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

NVCCFLAGS	:= -O3
NVCC		:= /usr/local/cuda/bin/nvcc
LD_LIBRARY_PATH	:= /usr/local/cuda/lib64


all: part1 part2 part3

part1: mp3-part1.cu mp3-util.h black_scholes.h black_scholes.cu
	$(NVCC) -DMP3_USE_DOUBLE_PRECISION -arch=sm_13 -o part1 mp3-part1.cu black_scholes.cu $(NVCCFLAGS)

# replace the previous with this build if your development machine is < SM 1.3 and does not support double precision
#part1: mp3-part1.cu mp3-util.h black_scholes.h black_scholes.cu
#	$(NVCC) -o part1 mp3-part1.cu black_scholes.cu $(NVCCFLAGS)


part2: mp3-part2.cu mp3-util.h compact.h compact.cu scan.h scan.cu scatter.h scatter.cu
	$(NVCC) -o part2 mp3-part2.cu compact.cu scan.cu scatter.cu $(NVCCFLAGS)

	
part3: mp3-part3.cu mp3-util.h black_scholes.h black_scholes.cu compact.h compact.cu scan.h scan.cu scatter.h scatter.cu
	$(NVCC) -DMP3_USE_DOUBLE_PRECISION -arch=sm_13 -o part3 mp3-part3.cu black_scholes.cu compact.cu scan.cu scatter.cu $(NVCCFLAGS)

# replace the previous with this build if your development machine is < SM 1.3 and does not support double precision
#part3: mp3-part3.cu mp3-util.h black_scholes.h black_scholes.cu compact.h compact.cu scan.h scan.cu scatter.h scatter.cu
#	$(NVCC) -o part3 mp3-part3.cu black_scholes.cu compact.cu scan.cu scatter.cu $(NVCCFLAGS)
	
clean:
	rm -rf part1 part2 part3

run1: part1
	./part1

run2: part2
	./part2

run3: part3
	./part3

