# Angle between vectors
Program to compute angle between two n-dimensional vectors using Nvidia CUDA API.

## Motivation

The goal of this program is to explore CUDA platform and optimize code to overcome some of its shortcomings.  For implementation details, execution timings and discussions please refer to the **project_document** file in the **doc** directory.

## Install

You will need to install CUDA Toolkit in order to compile and run the program. For details please refer to: 

[CUDA Toolkit Download](https://developer.nvidia.com/cuda-downloads).

After installing and performing all the necessary tweaks you can download and build the program with

```bash
$ git clone https://github.com/salmoor/angle-between-vectors.git
$ cd angle-between-vectors
$ make
```

## How to use

```bash
./a N blocksize [filename]
```

You can run the program as shown above where **N** is the number of integers each vector will have. **blocksize** is the number of threads within each block, for more details on CUDA's thread, block, and grid layout please refer to: [Thread Hierarchy](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#thread-hierarchy).

**filename** is optional, if given program will read integers from the input file. Format of the input file should be as follows:

```
N
1st integer of the vector 1
....
....
....
Nth integer of the vector 1
1st integer of the vector 2
....
....
....
Nth integer of the vector 2
```



If no file is specified, the program will generate random integers for you.

## Output

The program will then compute the angle between the two vectors using both serial single thread execution vs CUDA parallel execution and will output following information:

- Number of elements (integers in each array)
- Number of threads per block
- Number of blocks will be created
- Time for the array generation [ms]
- Time for the CPU function [ms]
- Time for the Host to Device transfer [ms]
- Time for the kernel execution [ms]
- Time for the Device to Host transfer [ms]
- Total execution time for GPU [ms]
- CPU result: [angle in degrees]
- GPU result: [angle in degrees]

