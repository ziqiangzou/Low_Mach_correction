# Euler_kokkos

## What is it ?

Provide performance portable Kokkos implementation for compressible
hydrodynamics.

## Dependencies

* [Kokkos](https://github.com/kokkos/kokkos)
* [CMake](https://cmake.org/) with version >= 3.1

## How to get Kokkos sources

Kokkos sources are included as a git submodule.

To download kokkos sources with project "euler_kokkos_two_phase" clone it with option "--recursive"
```
git clone --recursive https://github.com/ziqiangzou/euler_kokkos_two_phase.git
```

If you performed a regular "git clone", just type
```
git submodule init
git submodule update
```
to retrieve kokkos sources.

## Build

A few example builds

### Build without MPI / With Kokkos-openmp

Create a build directory, configure and make
```shell
mkdir build && cd build
cmake -DUSE_MPI=OFF -DKOKKOS_ENABLE_OPENMP=ON ..
make -j 4
```

Add variable CXX on the cmake command line to change the compiler
(clang++, icpc, pgcc, ....)

```shell
./euler_kokkos ./test.ini --kokkos-threads=4
```

## Developping with vim and youcomplete plugin

Assuming you are using vim (or neovim) text editor and have installed
the youcomplete plugin, you can have semantic autocompletion in a C++
project.

Make sure to have CMake variable CMAKE_EXPORT_COMPILE_COMMANDS set to
ON, and symlink the generated file to the top level source directory.

## Coding rules

For cmake follow [Effective Modern
CMake](https://gist.github.com/mbinna/c61dbb39bca0e4fb7d1f73b0d66a4fd1)
