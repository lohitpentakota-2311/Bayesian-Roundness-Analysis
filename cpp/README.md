\# C++ Implementation



This directory contains the C++17 implementation of selected functions from the

original MATLAB Bayesian Roundness Analysis workflow.



The C++ implementation is intended to reproduce the following MATLAB functions:



| C++ implementation | Original MATLAB function | Purpose |

|---|---|---|

| `cluster\_correlation.cpp/.hpp` | `clusterCorrelation` | Correlation-based variable clustering |

| `cross\_validation.cpp/.hpp` | `find\_lambda\_cv` | Group K-fold cross-validation for ridge regression |

| `ridge\_regression.cpp/.hpp` | `ridge\_model\_CI` | Ridge regression, prediction, uncertainty and confidence intervals |

| `hybrid\_error.cpp/.hpp` | `computeHybridrror` | Normalized hybrid prediction error |

| `test\_main.cpp` | — | Basic integration test |



\---



\## Requirements



\- C++17 compatible compiler

\- Eigen 3.x

\- Git (optional, for obtaining the repository)

\- MSYS2/MinGW-w64 can be used on Windows



The implementation uses the \[Eigen](https://eigen.tuxfamily.org/) library for

matrix and vector operations.



Eigen is a header-only library and does not need to be compiled separately.



\---



\## Directory Structure



```text

cpp/

├── cluster\_correlation.cpp

├── cluster\_correlation.hpp

├── cross\_validation.cpp

├── cross\_validation.hpp

├── hybrid\_error.cpp

├── hybrid\_error.hpp

├── ridge\_regression.cpp

├── ridge\_regression.hpp

├── test\_main.cpp

└── README.md

