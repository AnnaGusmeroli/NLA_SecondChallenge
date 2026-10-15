#include <iostream>
#include "utilities.hpp"
#include "lis_solver.hpp"

int main(int argc, char** argv) {
    const LIS_INT initialization_error = lis_initialize(&argc, &argv);
    if (initialization_error != LIS_SUCCESS) {
        std::cerr << "LIS initialization failed with error code "
                  << initialization_error << '\n';
        return 1;
    }
    
    /* Ex 1: Load the image as an Eigen matrix A with size m×n. Each entry in the matrix corresponds
    to a pixel on the screen and takes a value somewhere between 0 (black) and 255 (white).
    Compute the matrix product ATA and report the euclidean norm of ATA */
    std::cout << std::endl;
    std::cout << "Exercise 1: " << std::endl;
    Eigen::MatrixXd image = utilities::get_matrix_from_file("deer.jpg");
    std::cout << "Image size:" << image.rows() << " X " << image.cols() << std::endl;
    std::cout << "Non zero entries:" << image.nonZeros() << std::endl;
    Eigen::MatrixXd ATA = image.transpose() * image;
    std::cout << "Euclidean norm of ATA: " << ATA.norm() << std::endl << std::endl;

    /* Ex 2:Solve the eigenvalue problem ATAx= λx using the proper solver provided by the Eigen
    library. Report the two largest computed singular values of A. */
    std::cout << "Exercise 2: " << std::endl;
    
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saeigensolver(ATA);
    if (saeigensolver.info() != Eigen::Success) abort();
        Eigen::VectorXd singular_values = saeigensolver.eigenvalues().cwiseMax(0.0).array().sqrt().matrix();
        std::cout << "The two largest singular values of ATA are:\n" << singular_values.tail(2) << std::endl;
    
    /* Ex 3:Export matrix ATA in the matrix market format and move it to the lis-2.1.10_test
    folder. Using the proper iterative solver available in the LIS library compute the largest
    eigenvalue of ATA up to a tolerance of 10−10. Report the computed eigenvalue. Is the
    result in agreement with the one obtained in the previous point? */
    //LIS_INT err;
    const double tol=1e-10;
    auto lam1 = lis_solver::elis(ATA, tol, "pi", "ilu", 5000);
    std::cout << "Iterations: " << lam1.iterations << std::endl;
    std::cout << "  Residual: " << lam1.residual << std::endl;
    lis_finalize();
    
    const double sigma1_eigen = singular_values(singular_values.size() - 1);
    //in this case is not possible to use tail bc the result would be a size one vector and we wouldn't be able to confront the two values
    const double sigma1_lis = std::sqrt(lam1.eigenvalue.real()); //the result is complex althougt the imaginary part should be 0 since ATA is spd
    std::cout << "sigma1 is " << sigma1_eigen << " with saeigensolver" << std::endl;
    std::cout << "sigma1 is " << sigma1_lis << " with lis" << std::endl;
    if (sigma1_eigen-sigma1_lis <tol && std::abs(lam1.eigenvalue.imag()<tol)){
        std::cout << "the results are coerent" << std::endl << std::endl;
    }
    else{
        std::cout << "the results are not coerent" << std::endl << std::endl;
    }
    // Ex 4:
   
    // Ex 5:
    
    // Ex 6:
    

    // Ex 7:
    
    // Ex 8:
    
    


    // Ex 9:
    
    // Ex 10:
    

    // Ex 11:
    
    // Ex 12:
    
    // Ex 13:
    
    return 0;
}