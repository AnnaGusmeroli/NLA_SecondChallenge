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

    Result solve(
        const Eigen::SparseMatrix<double>& A_eigen,
        const Eigen::VectorXd& b_eigen,
        double tolerance,
        const std::string& solver_choice,
        const std::string& preconditioner_choice
    ) {
        //Check that the matrix is square
        if (A_eigen.rows() != A_eigen.cols()) {
            throw std::invalid_argument("A must be a square matrix.");
        }

        //Compatibility matrix vector check
        if (A_eigen.rows() != b_eigen.size()) {
            throw std::invalid_argument(
                "The dimensions of A and b do not match."
            );
        }

        const LIS_INT n = static_cast<LIS_INT>(A_eigen.rows());

        LIS_MATRIX A = nullptr;
        LIS_VECTOR b = nullptr;
        LIS_VECTOR x = nullptr;
        LIS_SOLVER solver = nullptr;

        check_error(
            lis_matrix_create(LIS_COMM_WORLD, &A),
            "lis_matrix_create"
        );

        check_error(
            lis_matrix_set_size(A, 0, n),
            "lis_matrix_set_size"
        );

        // Fill the LIS matrix with values from the Eigen sparse matrix
        for (int col = 0; col < A_eigen.outerSize(); ++col) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(A_eigen, col);
                 it;
                 ++it) {
                check_error(
                    lis_matrix_set_value(
                        LIS_INS_VALUE,
                        static_cast<LIS_INT>(it.row()),
                        static_cast<LIS_INT>(it.col()),
                        it.value(),
                        A
                    ),
                    "lis_matrix_set_value"
                );
            }
        }

        check_error(
            lis_matrix_set_type(A, LIS_MATRIX_CSR),
            "lis_matrix_set_type"
        );

        check_error(
            lis_matrix_assemble(A),
            "lis_matrix_assemble"
        );

        check_error(
            lis_vector_create(LIS_COMM_WORLD, &b),
            "lis_vector_create(b)"
        );

        check_error(
            lis_vector_set_size(b, 0, n),
            "lis_vector_set_size(b)"
        );

        // Fill the LIS vector with values from the Eigen vector
        for (LIS_INT i = 0; i < n; ++i) {
            check_error(
                lis_vector_set_value(
                    LIS_INS_VALUE,
                    i,
                    b_eigen(i),
                    b
                ),
                "lis_vector_set_value(b)"
            );
        }

        // Create the solution vector x and set its size, set all values to zero as an initial guess
        check_error(
            lis_vector_create(LIS_COMM_WORLD, &x),
            "lis_vector_create(x)"
        );

        check_error(
            lis_vector_set_size(x, 0, n),
            "lis_vector_set_size(x)"
        );

        check_error(
            lis_vector_set_all(0.0, x),
            "lis_vector_set_all(x)"
        );

        check_error(
            lis_solver_create(&solver),
            "lis_solver_create"
        );

        // Set solver options: BICGSTAB with ILU preconditioner and specified tolerance
        std::ostringstream options_stream;
        options_stream << "-i " << solver_choice << " -p " << preconditioner_choice
                       << std::setprecision(17)
                       << std::scientific
                       << tolerance;
        //if (preconditioner_choice == "ilu" && ilu_fill.has_value()) {
        //    options_stream << " -ilu_fill " << *ilu_fill;
        //}
        std::string options = options_stream.str();

        check_error(
            lis_solver_set_option(options.data(), solver),
            "lis_solver_set_option"
        );

        check_error(
            lis_solver_set_optionC(solver),
            "lis_solver_set_optionC"
        );

        // Solve the system Ax = b
        check_error(
            lis_solve(A, b, x, solver),
            "lis_solve"
        );

        LIS_INT iterations = 0;
        LIS_REAL residual = 0.0;

        // Get the number of iterations and the residual norm
        check_error(
            lis_solver_get_iter(solver, &iterations),
            "lis_solver_get_iter"
        );

        check_error(
            lis_solver_get_residualnorm(solver, &residual),
            "lis_solver_get_residualnorm"
        );

        // Extract the solution from the LIS vector
        Eigen::VectorXd solution(n);

        for (LIS_INT i = 0; i < n; ++i) {
            LIS_SCALAR value = 0.0;

            check_error(
                lis_vector_get_value(x, i, &value),
                "lis_vector_get_value"
            );

            solution(i) = value;
        }

        lis_solver_destroy(solver);
        lis_vector_destroy(x);
        lis_vector_destroy(b);
        lis_matrix_destroy(A);

        return {
            solution,
            iterations,
            residual
        };
    }
}
