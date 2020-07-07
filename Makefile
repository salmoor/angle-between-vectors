all: angle

angle: angle.cu Makefile
	nvcc -arch=sm_30 angle.cu -o a -lm