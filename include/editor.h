#pragma once

#include "io.h"

using namespace std;

typedef tuple<int, int, int> RGB;
typedef int Monochrome;

static RGB operator+(const RGB &l, const RGB &r);
static RGB operator*(const RGB &l, const RGB &r);
template<typename MonochromeType>
static RGB operator*(const RGB &l, const MonochromeType &r);
static std::ostream &operator << (std::ostream &out, const RGB &e);

vector<Image> split_image(const Image &im);
Image gaussian(const Image &im, double sigma, int radius);
Image gaussian_separable(const Image &im, double sigma, int radius);
Image sobel_x(const Image &im);
Image sobel_y(const Image &im);

template <typename KernelType>
static Image custom(const Image &im, const KernelType &kernel);

template<typename KernelType, typename ReturnType>
class ConvolutionFunctor
{
public:
	ConvolutionFunctor(const KernelType &k_);

	template<typename InputType>
	ReturnType operator()(const InputType &f);

private:
	const KernelType &kernel;
	const int diameter;
	const int row_shift;
	const int col_shift;
public:
	const int radius;
};

#include "editor_impl.h"