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
    
    // Ex 1: ciao
    
    std::cout << std::endl << "Exercise 1: " << std::endl;
    Eigen::MatrixXd A = utilities::get_matrix_from_file("deer.jpg");
    Eigen::MatrixXd AT_A = A.transpose() * A;
    std::cout << "The euclidean norm of AT * A is: " << AT_A.norm() << std::endl;


    // Ex 2:
    

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