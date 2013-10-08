#include "editor.h"

std::vector<Image> split_image(const Image &im)
{
	auto rows = im.n_rows / 3;
	auto cols = im.n_cols;

	return std::vector<Image>() = {
		im.submatrix(0, 0, rows, cols),
		im.submatrix(rows, 0, rows, cols),
		im.submatrix(2 * rows, 0, rows, cols),
	};
}

Image gaussian(const Image &im, double sigma, int radius)
{
	int n = radius * 2 + 1;
	Matrix<double> kernel(n, n);
	double k = 1 / (2 * M_PI * sigma * sigma);
	double d = -1 / (2 * sigma * sigma);
	double sum = 0;
	for (int i = 0; i < n; ++i)
		for (int j = 0; j < n; ++j) {
			kernel(i, j) = k * exp(d * (pow(i - radius, 2) + pow(j - radius, 2)));
			sum += kernel(i, j);
		}
	for (int i = 0; i < n; ++i)
		for (int j = 0; j < n; ++j)
			kernel(i, j) /= sum;
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j)
			cout << kernel(i, j) << " ";
		cout << endl;
	}
	return custom(im, kernel);
}

Image gaussian_separable(const Image &im, double sigma, int radius)
{
	double k = 1 / (2 * M_PI * sigma * sigma);
	double d = -1 / (2 * sigma * sigma);
	int n = radius * 2 + 1;
	Matrix<double> kernel_col(n, 1);
	Matrix<double> kernel_row(1, n);
	double sum = 0;
	for (int i = 0; i < n; ++i) {
		kernel_col(i, 0) = kernel_row(0, i) = k * exp(d * (pow(i - radius, 2)));
		sum += kernel_col(i, 0);
	}
	for (int i = 0; i < n; ++i) {
		kernel_col(i, 0) /= sum;
		kernel_row(0, i) /= sum;
	}
	return custom(custom(im, kernel_col), kernel_row);
}

Image sobel_x(const Image &im)
{
	Matrix<double> kernel = {
		{-1, 0, 1},
		{-2, 0, 2},
		{-1, 0, 1},
		// {-1, -2, -1},
		// {0, 0, 0},
		// {1, 2, 1},
	};
	return custom(im, kernel);
}

Image sobel_y(const Image &im)
{
	Matrix<double> kernel = {
		{1, 2, 1},
		{0, 0, 0},
		{-1, -2, -1},
		// {-1, 0, 1},
		// {-2, 0, 2},
		// {-1, 0, 1},
	};
	return custom(im, kernel);
}
