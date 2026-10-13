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
    Eigen::MatrixXd ATA = image.transpose() * image;
    std::cout << "Euclidean norm of ATA: " << ATA.norm() << std::endl << std::endl;

    /* Ex 2:Solve the eigenvalue problem ATAx= λxusing the proper solver provided by the Eigen
    library. Report the two largest computed singular values of A. */
    std::cout << "Exercise 2: " << std::endl;
    

    // Ex 3:
    std::cout << "Exercise 3: " << std::endl;
    Eigen::VectorXd v = utilities::convert_matrix_to_vector(image.cast<double>());
    Eigen::VectorXd w = utilities::convert_matrix_to_vector(noisy_image.cast<double>());
    std::cout << "image_vector v has " << v.size() << " components" << " = " << image.rows() << "rows * " << image.cols() << "cols = " << image.rows() * image.cols() << std::endl;
    std::cout << "noisy_image_vector w has " << w.size() << " components" << " = " << noisy_image.rows() << "rows * " << noisy_image.cols() << "cols = " << noisy_image.rows() * noisy_image.cols() << std::endl;
    std::cout << "Image vector v euclidean norm: " << v.norm() << std::endl << std::endl;

    // Ex 4:
    std::cout << "Exercise 4: " << std::endl;
    Eigen::MatrixXd Hav1(3, 3); //avarage smoothing filter
    Hav1 << 1./12., 1./12., 1./12.,
         1./12., 4./12., 1./12.,
         1./12., 1./12., 1./12.;

    Eigen::SparseMatrix<double> A1 = utilities::buildAconvolutionoperator(image.rows(), image.cols(), Hav1);
    std::cout << "Number of non zero elements of A1: " << A1.nonZeros() << std::endl;
    A1.prune(1.0,1e-15); //remove elements with absolute value less than 1e-15 to ensure that we are not counting almost 0 elements as non zero elements
    A1.makeCompressed();
    std::cout << "Number of non zero elements of A1 after pruning: " << A1.nonZeros() << std::endl << std::endl;

    // Ex 5:
    std::cout << "Exercise 5: " << std::endl;
    Eigen::VectorXd Blurred_noisy_image = A1 * w; // Apply the convolution operator A1 to the noisy image vector w
    //is not necessary to clamp the image because Hav1 sum to 1 and there is no negative values 
    Eigen::MatrixXd blurred_noisy_image_matrix = utilities::convert_vector_to_matrix(Blurred_noisy_image, image.rows(), image.cols());
    utilities::get_png_image_from_matrix(blurred_noisy_image_matrix, "blurred_noisy_deer.png");
    std::cout << "Saved blurred noisy image as 'blurred_noisy_deer.png'." << std::endl << std::endl;

    // Ex 6:
    std::cout << "Exercise 6: " << std::endl;
    Eigen::MatrixXd Hsh1(3, 3); //sharpening filter
    Hsh1 << 0, -3, 0,
         -1, 9, -3,
         0, -1, 0;
    Eigen::SparseMatrix<double> A2 = utilities::buildAconvolutionoperator(image.rows(), image.cols(), Hsh1);
    std::cout << "Number of non zero elements of A2: " << A2.nonZeros() << std::endl;
    A2.prune(1.0,1e-15);
    A2.makeCompressed();
    std::cout << "Number of non zero elements of A2 after pruning: " << A2.nonZeros() << std::endl;

    if (utilities::is_symmetric(A2)) {
        std::cout << "A2 is symmetric." << std::endl << std::endl;
    } else {
        std::cout << "A2 is not symmetric." << std::endl << std::endl;
    }

    // Ex 7:
    std::cout << "Exercise 7: " << std::endl;
    Eigen::VectorXd sharpened_image = A2 * v; // Apply the convolution operator A2 to the original image vector v
    sharpened_image = utilities::clamp_image(sharpened_image, 0, 255); // Clamp the sharpened image to ensure pixel values are within the valid range [0, 255]
    Eigen::MatrixXd sharpened_image_matrix = utilities::convert_vector_to_matrix(sharpened_image, image.rows(), image.cols());
    utilities::get_png_image_from_matrix(sharpened_image_matrix, "sharpened_deer.png");
    std::cout << "Saved sharpened image as 'sharpened_deer.png'." << std::endl << std::endl;

    // Ex 8:
    std::cout << "Exercise 8: " << std::endl; 
    Eigen::saveMarket(A2, "A2.mtx");
    Eigen::saveMarketVector(w, "w.mtx");
    // This passage was not necessary in our code because of how we implemented the lis_solver. We still did it because it was part of the exercise
    constexpr double tolerance = 1.0e-12;

    // Solve the linear system A2 * x = w using the lis_solver::solve function, that we introduced in the lis_solver.hpp file.
    // The function returns a lis_solver::Result struct that contains the solution vector, the number of iterations, and the residual.
    // It was implemented using the LIS library. In this way we can run main.cpp directly without having to compile the LIS library separately using .mtx files.
    lis_solver::Result result = lis_solver::solve(A2, w, tolerance); //Since A2 is not symmetric, we can use the Bicgstab method with ILU preconditioner

    std::cout << "Solved the linear system A2 * x = w using the bicgstab method with ILU preconditioner from the lis library." << std::endl;
    std::cout << "Tolerance: " << tolerance << std::endl;
    std::cout << "Iterations: " << result.iterations << std::endl;
    std::cout << "Residual: " << result.residual << std::endl << std::endl;

    lis_finalize();

    // Ex 9:
    std::cout << "Exercise 9: " << std::endl;
    Eigen::VectorXd x = result.solution;
    Eigen::VectorXd x_clamped = utilities::clamp_image(x, 0, 255); // Clamp the solution vector to ensure pixel values are within the valid range [0, 255]
    Eigen::MatrixXd x_image = utilities::convert_vector_to_matrix(x_clamped, image.rows(), image.cols());
    utilities::get_png_image_from_matrix(x_image, "x.png");
    std::cout << "Saved solution image as 'x.png'." << std::endl << std::endl;

    // Ex 10:
    std::cout << "Exercise 10: " << std::endl;
    Eigen::MatrixXd Hed2(3, 3); //Sobel filter for edge detection
    Hed2 << -1. ,0.,1.,
            -2. ,0.,2.,
            -1. ,0.,1.;

    Eigen::SparseMatrix<double> A3 = utilities::buildAconvolutionoperator(image.rows(), image.cols(), Hed2);
    std::cout << "Number of non zero elements of A3: " << A3.nonZeros() << std::endl;
    A3.prune(1.0,1e-15); //remove elements with absolute value less than 1e-15
    A3.makeCompressed();
    std::cout << "Number of non zero elements of A3 after pruning: " << A3.nonZeros() << std::endl;
    if (utilities::is_symmetric(A3)) {
        std::cout << "A3 is symmetric." << std::endl << std::endl;
    } else {
        std::cout << "A3 is not symmetric." << std::endl << std::endl;
    }

    // Ex 11:
    std::cout << "Exercise 11: " << std::endl;
    Eigen::VectorXd Edge_detection_image = A3 * v;
    Edge_detection_image = utilities::clamp_image(Edge_detection_image, 0, 255); // Clamp the edge detection image to ensure pixel values are within the valid range [0, 255]
    Eigen::MatrixXd Edge_detection_image_matrix = utilities::convert_vector_to_matrix(Edge_detection_image, image.rows(), image.cols());
    utilities::get_png_image_from_matrix(Edge_detection_image_matrix, "Edge_detection_deer.png");
    std::cout << "Saved edge detection image as 'Edge_detection_deer.png'." << std::endl << std::endl;

    // Ex 12:
    std::cout << "Exercise 12: " << std::endl;
    const Eigen::Index system_size = image.rows() * image.cols();
    Eigen::SparseMatrix<double> identity(system_size, system_size);
    identity.setIdentity(); //doing directy the sum without these two passeges resulted in nan bacause of the way set.Identity works
    Eigen::SparseMatrix<double> A_new = 4.0 * identity + A3;
    A_new.makeCompressed(); //makes sure the resulting matrix is sparse

    double tol = 1.e-10;                 // Convergence tolerance
    int maxit = 1000;           // Maximum iterations

    // Solving
    //since A3 is not symmetric and we only modified the diagonal, A_new is also not symmetric
    Eigen::BiCGSTAB<Eigen::SparseMatrix<double>, Eigen::DiagonalPreconditioner<double>> BiCGSTAB; //diag and not ilu cause computing ilu was too expansive to be worth it for this problem, and diag is enough to get a good solution
    BiCGSTAB.setMaxIterations(maxit);
    BiCGSTAB.setTolerance(tol);
    BiCGSTAB.compute(A_new);
    Eigen::VectorXd y = BiCGSTAB.solve(w);
    std::cout << "Solved the linear system (4I + A3) * y = w using the BiCGSTAB method with diagonal preconditioning from the Eigen library." << std::endl;
    std::cout << "Tolerance: " << tol << std::endl;
    std::cout << "#iterations: " << BiCGSTAB.iterations() << std::endl;
    std::cout << "relative residual: " << BiCGSTAB.error()      << std::endl << std::endl;

    // Ex 13:
    std::cout << "Exercise 13: " << std::endl; 
    Eigen::VectorXd y_clamped = utilities::clamp_image(y, 0, 255); //clamp the solution vector to ensure pixel values are within the valid range [0, 255]
    Eigen::MatrixXd y_image = utilities::convert_vector_to_matrix(y_clamped, image.rows(), image.cols());
    utilities::get_png_image_from_matrix(y_image, "y.png");
    std::cout << "Saved solution image as 'y.png'." << std::endl << std::endl;

    return 0;
}