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
    Eigen::MatrixXd ATA = image.transpose() * image;
    std::cout << "Euclidean norm of ATA: " << ATA.norm() << std::endl << std::endl;


    /* Ex 2:Solve the eigenvalue problem ATAx= λx using the proper solver provided by the Eigen
    library. Report the two largest computed singular values of A. */
    std::cout << "Exercise 2: " << std::endl;
    
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saeigensolver(ATA);
    if (saeigensolver.info() != Eigen::Success) abort();
        Eigen::VectorXd singular_values = saeigensolver.eigenvalues().cwiseMax(0.0).array().sqrt().matrix();
        std::cout << "The two largest singular values of A are:\n" << singular_values.tail(2) << std::endl;
    

    /* Ex 3:Export matrix ATA in the matrix market format and move it to the lis-2.1.10_test
    folder. Using the proper iterative solver available in the LIS library compute the largest
    eigenvalue of ATA up to a tolerance of 10−10. Report the computed eigenvalue. Is the
    result in agreement with the one obtained in the previous point? */
    std::cout << std::endl << "Exercise 3: " << std::endl;
    const double tol=1e-10;
    auto lam1 = lis_solver::elis(ATA, tol, "pi", "ilu", 5000); // This function exports the matrix in the matrix market format and then uses the lis library (etest1) to compute the largest eigenvalue of ATA
    std::cout << "Iterations: " << lam1.iterations << std::endl;
    std::cout << "Residual: " << lam1.residual << std::endl;
    std::cout << "Tolerance: " << tol << std::endl;

    const double lambda1_lis = lam1.eigenvalue.real(); //the result is complex althougt the imaginary part should be 0 since ATA is spd
    const double lambda1_eigen = saeigensolver.eigenvalues()(saeigensolver.eigenvalues().size() - 1); //in this case is not possible to use tail bc the result would be a size one vector and we wouldn't be able to confront the two values
    std::cout << "Largest eigenvalue of ATA with lis: " << lambda1_lis << std::endl;
    std::cout << "Largest eigenvalue of ATA with saeigensolver: " << lambda1_eigen << std::endl;
    std::cout << "Difference: " << std::abs(lambda1_lis - lambda1_eigen) << std::endl;
    if (std::abs(lambda1_eigen - lambda1_lis) / std::abs(lambda1_eigen) < tol &&
        std::abs(lam1.eigenvalue.imag()) < tol){
        std::cout << "the results are coerent" << std::endl;
    }
    else{
        std::cout << "the results are not coerent" << std::endl;
    }


    // Ex 4:
    std::cout << std::endl << "Exercise 4: " << std::endl;
    double shift = 0.99*lambda1_eigen; //We multiply by 0.99 to avoid the singularity of the matrix ATA - shift*I
    auto lam1_shift = lis_solver::elis(ATA, tol, "ii", "ilu", 5000, shift);
    std::cout << "Largest eigenvalue of ATA with shift " << shift << ": " << lam1_shift.eigenvalue.real() << std::endl;
    std::cout << "Iterations with shift: " << lam1_shift.iterations << std::endl;
    std::cout << "Residual with shift: " << lam1_shift.residual << std::endl;
    std::cout << "Tolerance with shift: " << tol << std::endl;

    std::cout << "Resisiduals difference: " << std::abs(lam1_shift.residual - lam1.residual) << std::endl;
    std::cout << "Iterations difference: " << std::abs(lam1_shift.iterations - lam1.iterations) << std::endl;
    std::cout << "Eigenvalue difference: " << std::abs(lam1_shift.eigenvalue.real() - lam1.eigenvalue.real()) << std::endl;


    // Ex 5:
    std::cout << std::endl << "Exercise 5: " << std::endl;
    std::cout << "Computing the SVD of the image matrix A using Eigen's BDCSVD." << std::endl;
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
    utilities::get_png_image_from_matrix(compressed_image_k1, "compressed_image_k1.png");
    std::cout << "Compressed image saved as compressed_image_k1.png" << std::endl;
    std::cout << "Compressing the image using the first k2 = " << k2 << " singular values and vectors." << std::endl;
    Eigen::MatrixXd compressed_image_k2 = C_k2 * D_k2.transpose();
    Eigen::VectorXd compressed_image_k2_vector = utilities::convert_matrix_to_vector(compressed_image_k2);
    compressed_image_k2_vector = utilities::clamp_image(compressed_image_k2_vector, 0, 255);
    compressed_image_k2 = utilities::convert_vector_to_matrix(compressed_image_k2_vector, compressed_image_k2.rows(), compressed_image_k2.cols());
    utilities::get_png_image_from_matrix(compressed_image_k2, "compressed_image_k2.png");
    std::cout << "Compressed image saved as compressed_image_k2.png" << std::endl;
    

    // Ex 8:
    std::cout << std::endl << "Exercise 8: " << std::endl;
    Eigen::MatrixXd checkerboard_image(400, 400);
    checkerboard_image.setZero();
    for (int i = 0; i < checkerboard_image.rows(); ++i) {
        for (int j = 0; j < checkerboard_image.cols(); ++j) {
            if ((i / 40 + j / 40) % 2 == 0) {
                checkerboard_image(i, j) = 0; // Black square
            } else {
                checkerboard_image(i, j) = 255;   // White square
            }
        }
    }
    Eigen::VectorXd checkerboard_vector = utilities::convert_matrix_to_vector(checkerboard_image);
    utilities::get_png_image_from_matrix(checkerboard_image, "checkerboard_image.png");
    std::cout << "Checkerboard image saved as checkerboard_image.png" << std::endl;
    std::cout << "The norm of the checkerboard image matrix is: " << checkerboard_image.norm() << std::endl;


    // Ex 9:
    std::cout << std::endl << "Exercise 9: " << std::endl;
    std::cout << "Adding noise to the checkerboard image." << std::endl;
    Eigen::MatrixXd noisy_checkerboard_image = utilities::add_noise_to_matrix(checkerboard_image, 60.0);
    Eigen::VectorXd noisy_checkerboard_vector = utilities::convert_matrix_to_vector(noisy_checkerboard_image);
    utilities::get_png_image_from_matrix(noisy_checkerboard_image, "noisy_checkerboard_image.png");
    std::cout << "Noisy checkerboard image saved as noisy_checkerboard_image.png" << std::endl;

    
    // Ex 10: Using the SVD module of the Eigen library, perform a singular value decomposition of the matrix corresponding to the noisy image. Report the two largest computed singular values.
    std::cout << std::endl << "Exercise 10: " << std::endl;
    std::cout << "Computing the SVD of the noisy image matrix using Eigen's BDCSVD." << std::endl;
    Eigen::BDCSVD<Eigen::MatrixXd> svd_noisy(noisy_checkerboard_image, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::VectorXd singular_values_noisy = svd_noisy.singularValues();

    Eigen::MatrixXd U_noisy = svd_noisy.matrixU();
    Eigen::MatrixXd V_noisy = svd_noisy.matrixV();
    Eigen::MatrixXd S_noisy = singular_values_noisy.asDiagonal();

    //singular values ordered in a decreasing manner
    std::cout << "The two largest singular values of the noisy image are:\n"
              << singular_values_noisy.head(2) << std::endl;


    // Ex 11: Starting from the previously computed SVD, create the matrices C and D defined in (1) assuming k = 6 and k = 12. Report the size of the matrices C and D.
    std::cout << std::endl << "Exercise 11: " << std::endl;
    int k3 = 6;
    int k4 = 12;
    Eigen::MatrixXd C_k3 = U_noisy.leftCols(k3);
    Eigen::MatrixXd D_k3 = V_noisy.leftCols(k3) * S_noisy.topLeftCorner(k3, k3);
    Eigen::MatrixXd C_k4 = U_noisy.leftCols(k4);
    Eigen::MatrixXd D_k4 = V_noisy.leftCols(k4) * S_noisy.topLeftCorner(k4, k4);
    std::cout << "Size of the matrix C_k3 (k = " << k3 << "): " << C_k3.rows() << " X " << C_k3.cols() << std::endl;
    std::cout << "Size of the matrix D_k3 (k = " << k3 << "): " << D_k3.rows() << " X " << D_k3.cols() << std::endl;
    std::cout << "Size of the matrix C_k4 (k = " << k4 << "): " << C_k4.rows() << " X " << C_k4.cols() << std::endl;
    std::cout << "Size of the matrix D_k4 (k = " << k4 << "): " << D_k4.rows() << " X " << D_k4.cols() << std::endl;


    // Ex 12: Compute the compressed images as the matrix product CD (again for k = 6 and k = 12). Export and upload the resulting images in .png.
    std::cout << std::endl << "Exercise 12: " << std::endl;
    std::cout << "Compressing the noisy image using the first k3 = " << k3 << " singular values and vectors." << std::endl;
    Eigen::MatrixXd compressed_noisy_k3 = C_k3 * D_k3.transpose();
    Eigen::VectorXd compressed_noisy_k3_vector = utilities::convert_matrix_to_vector(compressed_noisy_k3);
    compressed_noisy_k3_vector = utilities::clamp_image(compressed_noisy_k3_vector, 0, 255);
    compressed_noisy_k3 = utilities::convert_vector_to_matrix(compressed_noisy_k3_vector, compressed_noisy_k3.rows(), compressed_noisy_k3.cols());
    utilities::get_png_image_from_matrix(compressed_noisy_k3, "compressed_noisy_k3.png");
    std::cout << "Compressed image saved as compressed_noisy_k3.png" << std::endl;
    std::cout << "Compressing the noisy image using the first k4 = " << k4 << " singular values and vectors." << std::endl;
    Eigen::MatrixXd compressed_noisy_k4 = C_k4 * D_k4.transpose();
    Eigen::VectorXd compressed_noisy_k4_vector = utilities::convert_matrix_to_vector(compressed_noisy_k4);
    compressed_noisy_k4_vector = utilities::clamp_image(compressed_noisy_k4_vector, 0, 255);
    compressed_noisy_k4 = utilities::convert_vector_to_matrix(compressed_noisy_k4_vector, compressed_noisy_k4.rows(), compressed_noisy_k4.cols());
    utilities::get_png_image_from_matrix(compressed_noisy_k4, "compressed_noisy_k4.png");
    std::cout << "Compressed image saved as compressed_noisy_k4.png" << std::endl;


    // Ex 13: Compare the compressed images with the original and noisy images. Comment the results.
    std::cout << std::endl << "Exercise 13: " << std::endl;
    std::cout << "Comparing the compressed images with the original and noisy checkerboard." << std::endl;
    std::cout << "Relative error (k = " << k3 << ") with respect to the original: "
              << (compressed_noisy_k3 - checkerboard_image).norm() / checkerboard_image.norm() << std::endl;
    std::cout << "Relative error (k = " << k4 << ") with respect to the original: "
              << (compressed_noisy_k4 - checkerboard_image).norm() / checkerboard_image.norm() << std::endl;
    std::cout << "Relative error of the noisy image with respect to the original: "
              << (noisy_checkerboard_image - checkerboard_image).norm() / checkerboard_image.norm() << std::endl;
    
    std::cout << "COMMENT ON THE RESULTS: " << std::endl;
    std::cout << "It is very noticeable looking at the “deer image” that a lower value of " << std::endl;
    std::cout << "k means having more compression because less data is used. " << std::endl;
    std::cout << "In fact, the image with k=30 has a very bad quality and not many details. " << std::endl;
    std::cout << "This is also supported by our calculation: the matrices C and D have 19680 non-zero " << std::endl;
    std::cout << "entries each for k = 30 and 59040 each for k = 90, against the 430336 entries of A. " << std::endl;
    std::cout << "The effect on quality depends on the image. " << std::endl;
    std::cout << "The checkerboard is a very simple image with only two dominant singular values, " << std::endl;
    std::cout << "so a small k is enough to keep most details, instead the deer image is real photograph " << std::endl;
    std::cout << "and needs at least k=90. In the checkerboard image, it is observable that a low k " << std::endl;
    std::cout << "also reduces noise. The relative error with respect to the original goes from 0.137 " << std::endl;
    std::cout << "of the noisy image to 0.0873 with k = 6 and 0.0908 with k = 12. The k = 6 image " << std::endl;
    std::cout << "is slightly cleaner. Looking very carefully it is also noticeable that the black " << std::endl;
    std::cout << "squares in the compressed images look a bit less dark then in the original one, " << std::endl;
    std::cout << "that is because the interval [0, 255] reduces the contrast " << std::endl;
    std::cout << "(adding negative noise on black pixels and positive on white ones and then clamping in the range [0, 255] " << std::endl;
    std::cout << "shifts the average pixel values inward and reduces the gap between black and white) " << std::endl;
    std::cout << "and the SVD cannot recover the loss.  In conclusion, the truncated SVD method needs " << std::endl;
    std::cout << "to be used in a way that balances the compression of the images and " << std::endl;
    std::cout << "the detail preservation. " << std::endl;

    lis_finalize();
    
    return 0;
}