#pragma once

#include <cmath>
#include <iostream>
using namespace std;

template<typename KernelType, typename ReturnType>
ConvolutionFunctor<KernelType, ReturnType>\
	::ConvolutionFunctor(const KernelType &k_)
	:kernel(k_),
	 diameter(max(k_.n_rows, k_.n_cols)),
	 row_shift((diameter - k_.n_rows) / 2),
	 col_shift((diameter - k_.n_cols) / 2),
	 radius(diameter / 2)
{}

template<typename KernelType, typename ReturnType>
template<typename InputType>
ReturnType ConvolutionFunctor<KernelType, ReturnType>\
	::operator()(const InputType &f)
{
	ReturnType sum = ReturnType();
	for (int i = 0; i < kernel.n_rows; ++i)
		for (int j = 0; j < kernel.n_cols; ++j)
			sum = sum + (f(i + row_shift, j + col_shift) * kernel(i, j));
	return sum;
}

static RGB operator+(const RGB &l, const RGB &r)
{
	return RGB(
		get<0>(l) + get<0>(r),
		get<1>(l) + get<1>(r),
		get<2>(l) + get<2>(r)
	);
}

static RGB operator*(const RGB &l, const RGB &r)
{
	return RGB(
		get<0>(l) * get<0>(r),
		get<1>(l) * get<1>(r),
		get<2>(l) * get<2>(r)
	);
}

template<typename MonochromeType>
static RGB operator+(const RGB &l, const MonochromeType &r)
{
	return RGB(
		get<0>(l) * r,
		get<1>(l) * r,
		get<2>(l) * r
	);
}

template<typename MonochromeType>
static RGB operator*(const RGB &l, const MonochromeType &r)
{
	return RGB(
		get<0>(l) * r,
		get<1>(l) * r,
		get<2>(l) * r
	);
}

static bool operator<(const RGB &l, const Monochrome &r)
{
	return l < RGB(r, r, r);
}

static std::ostream &operator << (std::ostream &out, const RGB &e)
{
	return out << get<0>(e) << " " << get<1>(e) << " " << get<2>(e);
}

template <typename KernelType>
static Image custom(const Image &im, const KernelType &kernel)
{
	auto conv = ConvolutionFunctor<KernelType, RGB>(kernel);
	return im.unary_map(conv);
}

template<typename ValueType>
Matrix<ValueType> normalize(const Matrix<ValueType> &im, int t1 = 0, int t2 = 255)
{
    auto norm = NormalizeFunctor(t1, t2);
    return im.unary_map(norm);
}

template<typename ValueType>
std::vector<Matrix<ValueType>> split_image(const Matrix<ValueType> &im)
{
	auto rows = im.n_rows / 3;
	auto cols = im.n_cols;

	return std::vector<Matrix<ValueType>>({
		im.submatrix(0, 0, rows, cols),
		im.submatrix(rows, 0, rows, cols),
		im.submatrix(2 * rows, 0, rows, cols),
	});
}
