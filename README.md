# Numerical Linear Algebra — Challenge 2

Implementation of the second Numerical Linear Algebra challenge.

The program uses:

- [Eigen](https://eigen.tuxfamily.org/) for dense linear algebra and SVD;
- [LIS](https://www.ssisc.org/lis/) for iterative eigenvalue computations;
- `stb_image` and `stb_image_write` for reading and writing images.

## Students

- Francesco Barrale (10889288)
- Anna Adele Gusmeroli (10837456)
- Alessandra Mantovani (10809658)

## Overview

The program reads `deer.jpg` as a grayscale matrix and performs all 13
exercises from the challenge.

Pixel values are clamped to the range [0, 255] before being written as
grayscale PNG files.

## Requirements

The project is intended to run in the course environment with:

- GCC and `mpicxx`;
- Eigen 3;
- LIS;
- GNU Make.

Load the course environment before building:

```bash
source /u/sw/etc/bash.bashrc
module load gcc-glibc
module load lis
```

## Build and run

From the directory containing this README, run:

```bash
make
./main
```

The program expects `deer.jpg` in the project root. The output and solver
statistics are printed to the terminal.

To remove the executable and intermediate object files and rebuild from
scratch:

```bash
make clean
make
```

Additional build targets are available:

```bash
make run      # build, then run the program
make debug    # clean debug build
make release  # clean optimised build
```

## Generated files

The program creates the `png_files/` directory if it does not already exist.
The following grayscale PNG files are generated:

- `checkerboard_image.png` — original 400×400 checkerboard;
- `noisy_checkerboard_image.png` — checkerboard with random noise;
- `compressed_image_k1.png` — deer image compressed with \(k=30\);
- `compressed_image_k2.png` — deer image compressed with \(k=90\);
- `compressed_noisy_k3.png` — noisy checkerboard compressed with \(k=6\);
- `compressed_noisy_k4.png` — noisy checkerboard compressed with \(k=12\).

The LIS eigenvalue wrapper also writes the temporary Matrix Market file
`temp_matrix_elis.mtx` in the project root. It contains the sparse
representation of \(A^T A\) used by LIS.

## Project structure

```text
.
├── main.cpp                 # Entry point and challenge exercises
├── cpp_files/
│   ├── lis_solver.cpp       # LIS eigenvalue solver wrapper
│   └── utilities.cpp        # Image, noise, conversion, and PNG utilities
├── hpp_files/
│   ├── lis_solver.hpp       # LIS solver declarations
│   ├── utilities.hpp        # Utility declarations
│   ├── stb_image.h          # Image reader
│   └── stb_image_write.h    # Image writer
├── deer.jpg                 # Input grayscale image
├── Challenge2.pdf           # Challenge statement
├── Makefile
└── png_files/               # Generated PNG images
```

The `stb_image` and `stb_image_write` headers are course-provided image
utilities included in the project.
