#ifndef UTILITIES_HPP
#define UTILITIES_HPP

#include <Eigen/Dense>
#include <string>

// Function declarations
namespace utilities
{
    Eigen::MatrixXd get_matrix_from_file(const std::string& filename);

    Eigen::MatrixXd add_noise_to_matrix(const Eigen::MatrixXd& matrix, int noise_level);
    void get_png_image_from_matrix(const Eigen::MatrixXd& matrix, const std::string& filename);

    Eigen::VectorXd convert_matrix_to_vector(const Eigen::MatrixXd& matrix);
    Eigen::MatrixXd convert_vector_to_matrix(const Eigen::VectorXd& vector, int rows, int cols);

    Eigen::VectorXd clamp_image(Eigen::VectorXd& image_vector, int min_value, int max_value);
}

#endif // UTILITIES_HPP