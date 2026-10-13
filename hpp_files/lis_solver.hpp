#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <lis.h>

namespace lis_solver
{
    //result struct
    struct Result
    {
        Eigen::VectorXd solution;
        LIS_INT iterations;
        LIS_REAL residual;
    };

    void check_error(LIS_INT error, const char* function);

    Result solve(
        const Eigen::SparseMatrix<double>& A_eigen,
        const Eigen::VectorXd& b_eigen,
        double tolerance,
        const std::string& solver_choice
    );
}