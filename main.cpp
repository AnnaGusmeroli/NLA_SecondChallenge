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
    std::cout << std::endl << "Exercise 3: " << std::endl;
    const double tol=1e-10;
    auto lam1 = lis_solver::elis(ATA, tol, "pi", "ilu", 5000);
    std::cout << "Iterations: " << lam1.iterations << std::endl;
    std::cout << "  Residual: " << lam1.residual << std::endl;
    
    const double sigma1_eigen = singular_values(singular_values.size() - 1);
    //in this case is not possible to use tail bc the result would be a size one vector and we wouldn't be able to confront the two values
    const double sigma1_lis = std::sqrt(lam1.eigenvalue.real()); //the result is complex althougt the imaginary part should be 0 since ATA is spd
    std::cout << "sigma1 is " << sigma1_eigen << " with saeigensolver" << std::endl;
    std::cout << "sigma1 is " << sigma1_lis << " with lis" << std::endl;
    if (std::abs(sigma1_eigen - sigma1_lis) < tol &&
        std::abs(lam1.eigenvalue.imag()) < tol){
        std::cout << "the results are coerent" << std::endl;
    }
    else{
        std::cout << "the results are not coerent" << std::endl;
    }


    // Ex 4:
    std::cout << std::endl << "Exercise 4: " << std::endl;
    double shift = sigma1_eigen*sigma1_eigen - 1000.0; 
    auto lam1_shift = lis_solver::elis(ATA, tol, "si", "ilu", 5000, shift);
    std::cout << "Largest eigenvalue of ATA with shift " << shift << ": " << lam1_shift.eigenvalue.real() << std::endl;
    std::cout << "Iterations: " << lam1_shift.iterations << std::endl;
    std::cout << "  Residual: " << lam1_shift.residual << std::endl;


    // Ex 5:
    std::cout << std::endl << "Exercise 5: " << std::endl;
    std::cout << "Computing the SVD of the image matrix A using Eigen's JacobiSVD." << std::endl;
    Eigen::BDCSVD<Eigen::MatrixXd> svd(image, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::VectorXd singular_values_svd = svd.singularValues();

    Eigen::MatrixXd U = svd.matrixU();
    Eigen::MatrixXd V = svd.matrixV();
    Eigen::MatrixXd S = singular_values_svd.asDiagonal();

    std::cout << "Norm of the matrix Sigma: " << S.norm() << std::endl;
    

    // Ex 6:
    std::cout << std::endl << "Exercise 6: " << std::endl;
    int k1 = 30;
    int k2 = 90;
    Eigen::MatrixXd C_k1 = U.leftCols(k1);
    Eigen::MatrixXd D_k1= V.leftCols(k1)  * S.topLeftCorner(k1, k1);
    Eigen::MatrixXd C_k2 = U.leftCols(k2);
    Eigen::MatrixXd D_k2= V.leftCols(k2)  * S.topLeftCorner(k2, k2);
    std::cout << "Number of non-zero entries of the matrix C_k1: " << C_k1.count() << std::endl;
    std::cout << "Number of non-zero entries of the matrix D_k1: " << D_k1.count() << std::endl;
    std::cout << "Number of non-zero entries of the matrix C_k2: " << C_k2.count() << std::endl;
    std::cout << "Number of non-zero entries of the matrix D_k2: " << D_k2.count() << std::endl;

    // Ex 7:
    std::cout << std::endl << "Exercise 7: " << std::endl;
    std::cout << "Compressing the image using the first k1 = " << k1 << " singular values and vectors." << std::endl;
    Eigen::MatrixXd compressed_image_k1 = C_k1 * D_k1.transpose();
    Eigen::VectorXd compressed_image_k1_vector = utilities::convert_matrix_to_vector(compressed_image_k1);
    compressed_image_k1_vector = utilities::clamp_image(compressed_image_k1_vector, 0, 255);
    compressed_image_k1 = utilities::convert_vector_to_matrix(compressed_image_k1_vector, compressed_image_k1.rows(), compressed_image_k1.cols());
    utilities::get_png_image_from_matrix(compressed_image_k1, "compressed_image_k1.jpg");
    std::cout << "Compressed image saved as compressed_image_k1.jpg" << std::endl;
    std::cout << "Compressing the image using the first k2 = " << k2 << " singular values and vectors." << std::endl;
    Eigen::MatrixXd compressed_image_k2 = C_k2 * D_k2.transpose();
    Eigen::VectorXd compressed_image_k2_vector = utilities::convert_matrix_to_vector(compressed_image_k2);
    compressed_image_k2_vector = utilities::clamp_image(compressed_image_k2_vector, 0, 255);
    compressed_image_k2 = utilities::convert_vector_to_matrix(compressed_image_k2_vector, compressed_image_k2.rows(), compressed_image_k2.cols());
    utilities::get_png_image_from_matrix(compressed_image_k2, "compressed_image_k2.jpg");
    std::cout << "Compressed image saved as compressed_image_k2.jpg" << std::endl;
    
    // Ex 8:
    
    


    // Ex 9:
    
    // Ex 10:
    

    // Ex 11:
    
    // Ex 12:
    
    // Ex 13:

    lis_finalize();
    
    return 0;
}