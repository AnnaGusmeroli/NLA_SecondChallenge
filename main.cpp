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
    
    // Ex 3:
    
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