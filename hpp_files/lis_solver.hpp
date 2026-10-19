#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <lis.h>
#include <complex>
#include <string>
#include <unsupported/Eigen/SparseExtra>

namespace lis_solver
{
    // Result struct for eigenvalues
    struct EigenvalueResult
    {
        std::complex<double> eigenvalue;
        Eigen::VectorXd eigenvector;
        LIS_INT iterations;
        LIS_REAL residual;
    };

    void check_error(LIS_INT error, const char* function);

    // LIS function to find eigenpairs
    EigenvalueResult elis(
        const Eigen::MatrixXd& A_eigen,
        double tolerance,
        const std::string& esolver_choice,
        const std::string& preconditioner_choice,
        int max_iter = 1000,
        double shift = 0.0
    );
}