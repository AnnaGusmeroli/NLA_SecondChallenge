#include "lis_solver.hpp"

#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

namespace lis_solver
{
    void check_error(LIS_INT error, const char* function)
    {
        if (error != LIS_SUCCESS) {
            throw std::runtime_error(
                std::string(function) +
                " failed with error code " +
                std::to_string(error)
            );
        }
    }

    EigenvalueResult elis(
        const Eigen::MatrixXd& A_eigen,
        double tolerance,
        const std::string& esolver_choice,
        const std::string& preconditioner_choice,
        int max_iter,
        double shift
    )
    {
        std::string temp_matrix_filename = "temp_matrix_elis.mtx";
        Eigen::SparseMatrix<double> sparse_A = A_eigen.sparseView();
        Eigen::saveMarket(sparse_A, temp_matrix_filename);

        LIS_MATRIX A;
        LIS_VECTOR x;
        LIS_SCALAR evalue0;
        LIS_ESOLVER esolver;
        LIS_INT err;
        LIS_INT nesol;
        LIS_REAL residual;
        LIS_INT iter;

        err = lis_matrix_create(LIS_COMM_WORLD, &A);
        check_error(err, "lis_matrix_create");

        err = lis_input_matrix(A, (char*)temp_matrix_filename.c_str());
        check_error(err, "lis_input_matrix");

        err = lis_vector_duplicate(A, &x);
        check_error(err, "lis_vector_duplicate");

        err = lis_esolver_create(&esolver);
        check_error(err, "lis_esolver_create");

        std::ostringstream options_stream;
        options_stream << "-e " << esolver_choice
                       << " -p " << preconditioner_choice
                       << std::setprecision(17)
                       << std::scientific
                       << " -etol " << tolerance
                       << " -emaxiter " << max_iter
                       << " -shift " << shift;

        std::string opt_str = options_stream.str();
        err = lis_esolver_set_option(const_cast<char*>(opt_str.c_str()), esolver);
        check_error(err, "lis_esolver_set_option");

        err = lis_esolve(A, x, &evalue0, esolver);
        check_error(err, "lis_esolve");

        err = lis_esolver_get_esolver(esolver, &nesol);
        check_error(err, "lis_esolver_get_esolver");

        err = lis_esolver_get_residualnorm(esolver, &residual);
        check_error(err, "lis_esolver_get_residualnorm");

        err = lis_esolver_get_iter(esolver, &iter);
        check_error(err, "lis_esolver_get_iter");

        LIS_INT n = A->n;
        Eigen::VectorXd eigenvector_eigen(n);
        for (LIS_INT i = 0; i < n; ++i)
        {
            LIS_SCALAR val;
            lis_vector_get_value(x, i, &val);
            eigenvector_eigen[i] = static_cast<double>(val);
        }

        EigenvalueResult result;
        result.eigenvalue = evalue0;
        result.eigenvector = eigenvector_eigen;
        result.iterations = iter;
        result.residual = residual;

        lis_esolver_destroy(esolver);
        lis_matrix_destroy(A);
        lis_vector_destroy(x);

        return result;
    }
}