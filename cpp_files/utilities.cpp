#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#endif

#include "utilities.hpp"
#include "stb_image.h"
#include "stb_image_write.h"

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

#include <filesystem>
#include <algorithm>
#include <iostream>
#include <random>

// Function definitions
namespace utilities
{
    // Function to read an image file and return it as an Eigen matrix
    Eigen::MatrixXd get_matrix_from_file(const std::string& filename)
    {
        int width;
        int height;
        int channels;

        unsigned char* image = stbi_load(
            filename.c_str(),
            &width,
            &height,
            &channels,
            1 
        );

        if (!image)
        {
            std::cerr << "Error: " << stbi_failure_reason() << std::endl;
            return Eigen::MatrixXd();
        }

        // Create an Eigen matrix from the loaded image
        Eigen::MatrixXd matrix(height, width);
        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                matrix(i, j) = image[i * width + j];
            }
        }

        stbi_image_free(image);
        return matrix;
    }

    // Function to add noise to an Eigen matrix
    Eigen::MatrixXd add_noise_to_matrix(const Eigen::MatrixXd& matrix, int noise_level)
    {
        static std::mt19937 generator(0); // Fixed seed for reproducibility
        std::uniform_int_distribution<int> distribution(-noise_level, noise_level);
        Eigen::MatrixXd noisy_matrix = matrix;

        for (int i = 0; i < noisy_matrix.rows(); ++i)
        {
            for (int j = 0; j < noisy_matrix.cols(); ++j)
            {   
                // Add noise to the current element, keeping the value within the range [0, 255]
                int noisy_value = noisy_matrix(i, j) + distribution(generator);
                noisy_matrix(i, j) = std::clamp(noisy_value, 0, 255);
            }
        }
        return noisy_matrix;
    }

    // Function to save an Eigen matrix as a PNG image
    void get_png_image_from_matrix(const Eigen::MatrixXd& matrix, const std::string& filename)
    {
        int width = matrix.cols();
        int height = matrix.rows();
        unsigned char* image = new unsigned char[width * height];

        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                image[i * width + j] = static_cast<unsigned char>(matrix(i, j));
            }
        }

        // Create the output directory if it doesn't exist
        const std::filesystem::path output_directory = "png_files";
        std::filesystem::create_directories(output_directory);
        const std::filesystem::path output_path = output_directory / filename;

        if (stbi_write_png(output_path.string().c_str(), width, height, 1, image, width) == 0)
        {
            std::cerr << "Error: could not write PNG file "
                      << output_path << std::endl;
        }
        delete[] image;
    }

    // Function to convert an Eigen matrix to a vector
    Eigen::VectorXd convert_matrix_to_vector(const Eigen::MatrixXd& matrix)
    {
        Eigen::VectorXd vector(matrix.size());
        for (int i = 0; i < matrix.rows(); ++i)
        {
            for (int j = 0; j < matrix.cols(); ++j)
            {
                vector(i * matrix.cols() + j) = matrix(i, j);
            }
        }
        return vector;
    }

    // Function to convert a vector back to an Eigen matrix
    Eigen::MatrixXd convert_vector_to_matrix(const Eigen::VectorXd& vector, int rows, int cols)
    {
        if (vector.size() != rows * cols)
        {
            std::cerr << "Error: Vector size does not match the specified matrix dimensions." << std::endl;
            return Eigen::MatrixXd(); // Return an empty matrix if dimensions do not match
        }

        Eigen::MatrixXd matrix(rows, cols);
        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < cols; ++j)
            {
                matrix(i, j) = vector(i * cols + j);
            }
        }
        return matrix;
    }

    // Function to clamp the values of an Eigen vector within a specified range [min_value, max_value]
    Eigen::VectorXd clamp_image(Eigen::VectorXd& image_vector, int min_value, int max_value)
    {
        Eigen::VectorXd clamped_vector = image_vector;
        for (int i = 0; i < clamped_vector.size(); ++i)
        {
            clamped_vector(i) = std::clamp(clamped_vector(i), static_cast<double>(min_value), static_cast<double>(max_value));
        }
        return clamped_vector;
    }

}