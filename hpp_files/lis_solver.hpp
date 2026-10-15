#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <lis.h>
#include <optional>
#include <complex>
#include <string>
#include <iostream>
#include <unsupported/Eigen/SparseExtra>

namespace lis_solver
{
    // Result struct for linear system
    struct Result
    {
        Eigen::VectorXd solution;
        LIS_INT iterations;
        LIS_REAL residual;
    };

    // Result struct for eigenvalues
    struct EigenvalueResult
    {
        std::complex<double> eigenvalue;
        Eigen::VectorXd eigenvector;
        LIS_INT iterations;
        LIS_REAL residual;
    };

    void check_error(LIS_INT error, const char* function);

    // LIS solver for linear sistems
    Result solve(
        const Eigen::SparseMatrix<double>& A_eigen,
        const Eigen::VectorXd& b_eigen,
        double tolerance,
        const std::string& solver_choice,
        const std::string& preconditioner_choice //,
        //std::optional<int> ilu_fill = std::nullopt
    );

    // LIS function to find eigenpairs
    EigenvalueResult elis(
        const Eigen::MatrixXd& A_eigen,
        double tolerance,
        const std::string& esolver_choice,
        const std::string& preconditioner_choice,
        int max_iter = 1000
    );
}