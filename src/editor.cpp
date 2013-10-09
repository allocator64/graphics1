#include "editor.h"

#include <queue>
#include <cassert>

template<typename ValueType>
static inline ValueType pow_2(ValueType val)
{
	return val * val;
}

// static inline RGB pow(const RGB &l, double r)
// {
// 	return RGB(
// 		pow(get<0>(l), r),
// 		pow(get<1>(l), r),
// 		pow(get<2>(l), r)
// 	);
// }

Matrix<Monochrome> ImageToMonochrome(const Image &im)
{
	Matrix<Monochrome> result(im.n_rows, im.n_cols);
	for (int i = 0; i < im.n_rows; ++i)
		for (int j = 0; j < im.n_cols; ++j)
			result(i, j) = get<0>(im(i, j));
	return result;
}

Image MonochromeToImage(const Matrix<Monochrome> &im)
{
	Image result(im.n_rows, im.n_cols);
	for (int i = 0; i < im.n_rows; ++i)
		for (int j = 0; j < im.n_cols; ++j)
			result(i, j) = RGB(
				im(i, j),
				im(i, j),
				im(i, j)
			);
	return result;
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
			kernel(i, j) = k * exp(d * (pow_2(i - radius) + pow_2(j - radius)));
			sum += kernel(i, j);
		}
	for (int i = 0; i < n; ++i)
		for (int j = 0; j < n; ++j)
			kernel(i, j) /= sum;
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
		kernel_col(i, 0) = kernel_row(0, i) = k * exp(d * (pow_2(i - radius)));
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
	Matrix<Monochrome> kernel = {
		{-1, 0, 1},
		{-2, 0, 2},
		{-1, 0, 1},
	};
	return custom(im, kernel);
}

Image sobel_y(const Image &im)
{
	Matrix<Monochrome> kernel = {
		{1, 2, 1},
		{0, 0, 0},
		{-1, -2, -1},
	};
	return custom(im, kernel);
}

class AbsGradientFunctor
{
public:
	template<typename InputType>
	Monochrome operator()(const InputType &left, const InputType &right) const
	{
		return pow(pow_2(left(0, 0)) + pow_2(right(0, 0)), 0.5);
	}
	static const int radius = 0;
};

class DirectGradientFunctor
{
public:
	template<typename InputType>
	double operator()(const InputType &left, const InputType &right) const
	{
		return atan2(right(0, 0), left(0, 0));
	}
	static const int radius = 0;
};

class SuppressionFunctor
{
public:
	template<typename InputType, typename InputType2>
	Monochrome operator()(const InputType &abs, const InputType2 &direct) const
	{
		int i = lower_bound(mas.begin(), mas.end(), direct(1, 1)) - mas.begin();
		int first = (num[i] + 2) % 8;
		int second = (first + 4) % 8;
		if (
			abs(1 + di[first], 1 + dj[first]) > abs(1, 1) ||
			abs(1 + di[second], 1 + dj[second]) > abs(1, 1)
		)
			return 0;
		return abs(1, 1);
	}
	static const int radius = 1;
private:
	static const vector<double> mas;
	static const vector<int> num;
	static const vector<int> di;
	static const vector<int> dj;
};
static const double pp = M_PI / 8;
const vector<double> SuppressionFunctor::mas = {-7*pp, -5*pp, -3*pp, -pp, pp, 3*pp, 5*pp, 7*pp, 8*pp};
const vector<int> SuppressionFunctor::num    = {   4,    5,    6,  7, 0,   1,   2,   3,   4};
const vector<int> SuppressionFunctor::di = {-1, -1, 0, 1, 1, 1, 0, -1};
const vector<int> SuppressionFunctor::dj = {0, 1, 1, 1, 0, -1, -1, -1};

class TreshholdFunctor
{
public:
	TreshholdFunctor(int min_, int max_)
		: t_min(min_), t_max(max_)
	{}
	template<typename InputType>
	Monochrome operator()(const InputType &im) const
	{
		if (im(0, 0) < t_min)
			return 0;
		if (im(0, 0) > t_max)
			return 255;
		return 128;
	}
	static const int radius = 0;
	const int t_min;
	const int t_max;	
};

template <typename InputType>
void hysteresis_bfs(InputType &abs_grad)
{
	const vector<int> di = {0, 0, 1, -1};
	const vector<int> dj = {1, -1, 0, 0};
	auto exist = [&](int i, int j){ return 0 <= i && i < abs_grad.n_rows && 0 <= j && j < abs_grad.n_cols;};
	vector<vector<char>> visited(abs_grad.n_rows, vector<char>(abs_grad.n_cols, 0));
	for (int i = 0; i < abs_grad.n_rows; ++i)
		for (int j = 0; j < abs_grad.n_cols; ++j)
			if (!visited[i][j] && abs_grad(i, j) == 255) {
				queue<pair<int, int>> Q;
				Q.push(make_pair(i, j));
				visited[i][j] = 1;
				abs_grad(i, j) = 255;
				while (!Q.empty()) {
					int ti = Q.front().first;
					int tj = Q.front().second;
					Q.pop();
					for (int k = 0; k < 4; ++k) {
						int ni = ti + di[k];
						int nj = tj + dj[k];
						if (
							exist(ni, nj) &&
							!visited[ni][nj] &&
							abs_grad(ni, nj) >= 128
						) {
							visited[ni][nj] = 1;
							abs_grad(ni, nj) = 255;
							Q.push(make_pair(ni, nj));
						}
					}
				}
			}
	for (int i = 0; i < abs_grad.n_rows; ++i)
		for (int j = 0; j < abs_grad.n_cols; ++j)
			if (abs_grad(i, j) == 128)
				abs_grad(i, j) = 0;
}

class DSU
{
public:
	DSU(int n)
		: parent(n), rank(n), marks(n)
	{}

	void make_set(int v)
	{
		parent[v] = v;
		rank[v] = 0;
		marks[v] = 0;
	}
	
	void set_mark(int v)
	{
		marks[find_set(v)] = 1;
	}

	bool check_mark(int v)
	{
		return marks[find_set(v)];
	}

	int find_set(int v)
	{
		if (parent[v] == v)
			return v;
		return parent[v] = find_set(parent[v]);
	}
	 
	int union_sets(int a, int b)
	{
		a = find_set(a);
		b = find_set(b);
		if (a != b) {
			if (rank[a] < rank[b])
				swap(a, b);
			parent[b] = a;
			marks[a] = (marks[a] | marks[b]);
			if (rank[a] == rank[b])
				++rank[a];
		}
		return a;
	}

private:
	vector<int> parent;
	vector<int> rank;
	vector<char> marks;
};

template <typename InputType>
void hysteresis_dsu(InputType &abs_grad)
{
	int len = abs_grad.n_rows * abs_grad.n_cols;
	auto dsu = DSU(len);
	auto exist = [&](int i, int j){ return 0 <= i && i < abs_grad.n_rows && 0 <= j && j < abs_grad.n_cols;};
	auto check = [&](int i, int j){ return exist(i, j) && (abs_grad(i, j) >= 128);};
	auto strong = [&](int i, int j){ return exist(i, j) && (abs_grad(i, j) == 255);};
	auto f = [&](int i, int j){ return i * abs_grad.n_cols + j;};
	vector<vector<int>> groups(abs_grad.n_rows, vector<int>(abs_grad.n_cols));
	int count = 0;
	for (int i = 0; i < abs_grad.n_rows; ++i)
		for (int j = 0; j < abs_grad.n_cols; ++j) {
			bool A = check(i, j);
			bool C = check(i - 1, j);
			bool B = check(i, j - 1);
			if (!A)
				continue;
			int current;
			if (!B && !C) {
				current = count++;
				dsu.make_set(current);
			}

			if (B && !C)
				current = dsu.find_set(groups[i][j - 1]);
			if (!B && C)
				current = dsu.find_set(groups[i - 1][j]);
			if (B && C)
				current = dsu.union_sets(groups[i - 1][j], groups[i][j - 1]);

			if (strong(i, j))
				dsu.set_mark(current);

			groups[i][j] = current;
		}
	for (int i = 0; i < abs_grad.n_rows; ++i)
		for (int j = 0; j < abs_grad.n_cols; ++j)
			abs_grad(i, j) = 255 * (check(i, j) && dsu.check_mark(groups[i][j]));
}

Image canny(const Image &im, int threshold1, int threshold2)
{
	auto blured = gaussian_separable(im, 1.4, 5);
	auto derivative_x = ImageToMonochrome(sobel_x(blured));
	auto derivative_y = ImageToMonochrome(sobel_y(blured));

	auto abs_grad_functor = AbsGradientFunctor();
	auto abs_grad = binary_map(abs_grad_functor, derivative_x, derivative_y);

	auto direct_grad_functor = DirectGradientFunctor();
	auto direct_grad = binary_map(direct_grad_functor, derivative_x, derivative_y);

	auto suppression_functor = SuppressionFunctor();
	abs_grad = binary_map(suppression_functor, abs_grad, direct_grad);

	auto treshold_functor = TreshholdFunctor(threshold1, threshold2);
	abs_grad = abs_grad.unary_map(treshold_functor);

	hysteresis_dsu(abs_grad);

	auto result = MonochromeToImage(normalize(abs_grad));
	return result;
}

template<typename MetricType>
static void match(MonochromeImage &r, MonochromeImage &g, MetricType metric)
{
	vector<pair<double, pair<int, int>>> v;
	for (int di = -15; di <= 15; ++di)
		for (int dj = -15; dj <= 15; ++dj) {
			MonochromeImage r_ = r.submatrix(max(0, di), max(0, dj), min(r.n_rows, g.n_rows + di) - max(0, di), min(r.n_cols, g.n_cols + dj) - max(0, dj));
			MonochromeImage g_ = g.submatrix(max(0, -di), max(0, -dj), min(g.n_rows, r.n_rows - di) - max(0, -di), min(g.n_cols, r.n_cols - dj) - max(0, -dj));
			assert(r_.n_cols == g_.n_cols);
			assert(r_.n_rows == g_.n_rows);
			v.push_back(make_pair(metric(r_, g_), make_pair(di, dj)));
		}
	sort(v.begin(), v.end());
	int row_shift = v[0].second.first;
	int col_shift = v[0].second.second;
	MonochromeImage r_ = r.submatrix(max(0, row_shift), max(0, col_shift), min(r.n_rows, g.n_rows + row_shift) - max(0, row_shift), min(r.n_cols, g.n_cols + col_shift) - max(0, col_shift));
	MonochromeImage g_ = g.submatrix(max(0, -row_shift), max(0, -col_shift), min(g.n_rows, r.n_rows - row_shift) - max(0, -row_shift), min(g.n_cols, r.n_cols - col_shift) - max(0, -col_shift));
	r = r_;
	g = g_;
}

Image align(const Image &image, const string &postprocessing)
{
	auto tmp = split_image(image);
	for (auto &im : tmp) {
		auto borders = ImageToMonochrome(canny(im, 36, 100));
		vector<int> sum_row(borders.n_rows);
		vector<int> sum_col(borders.n_cols);
		for (int i = 0; i < borders.n_rows; ++i)
			for (int j = 0; j < borders.n_cols; ++j) {
				sum_row[i] += (borders(i, j) != 0);
				sum_col[j] += (borders(i, j) != 0);
			}

		vector<pair<int, int>> v;
		for (int i = 0; i < borders.n_rows * 0.05; ++i)
			v.push_back(make_pair(sum_row[i], i));
		sort(v.rbegin(), v.rend());
		int row1 = max(v[0].second, v[1].second) + 1;

		v.clear();
		for (int i = 0; i < borders.n_cols * 0.05; ++i)
			v.push_back(make_pair(sum_col[i], i));
		sort(v.rbegin(), v.rend());
		int col1 = max(v[0].second, v[1].second) + 1;

		v.clear();
		for (int i = borders.n_rows * 0.95; i < borders.n_rows; ++i)
			v.push_back(make_pair(sum_row[i], i));
		sort(v.rbegin(), v.rend());
		int row2 = min(v[0].second, v[1].second);

		v.clear();
		for (int i = borders.n_cols * 0.95; i < borders.n_cols; ++i)
			v.push_back(make_pair(sum_col[i], i));
		sort(v.rbegin(), v.rend());
		int col2 = min(v[0].second, v[1].second);

		im = im.submatrix(row1, col1, row2 - row1, col2 - col1);
	}
	vector<MonochromeImage> rgb(3);
	for (int i = 0; i < 3; ++i)
		rgb[i] = ImageToMonochrome(tmp[i]);
	MonochromeImage &r = rgb[0];
	MonochromeImage &g = rgb[1];
	MonochromeImage &b = rgb[2];

	auto mse = [](MonochromeImage &im1, MonochromeImage &im2) {
		double sum = 0;
		int row_lim = min(im1.n_rows, im2.n_rows);
		int col_lim = min(im1.n_cols, im2.n_cols);
		for (int i = 0; i < row_lim; ++i)
			for (int j = 0; j < col_lim; ++j)
				sum += pow_2(im1(i, j) - im2(i, j));
		sum /= (row_lim * col_lim);
		return sum;
	};

	auto cc = [](MonochromeImage &im1, MonochromeImage &im2) {
		double sum = 0;
		for (int i = 0; i < im1.n_rows; ++i)
			for (int j = 0; j < im2.n_cols; ++j)
				sum += im1(i, j) * im2(i, j);
		sum *= -1;
		return sum;
	};

	match(r, g, mse);
	match(r, b, mse);
	match(b, g, mse);

	auto result = MonochromeToImage(r);

	auto safe_get = [](MonochromeImage &im, int i, int j) { return (i < im.n_rows && j < im.n_cols) ? im(i, j) : 0;};

	for (int i = 0; i < result.n_rows; ++i)
		for (int j = 0; j < result.n_cols; ++j)
			result(i, j) = RGB(
				safe_get(b, i, j), 
				safe_get(g, i, j), 
				safe_get(r, i, j)
			);
	
	if (postprocessing == "--gray-world")
		result = gray_world(result);
	else if (postprocessing == "--unsharp")
		result = unsharp(result);
	return result;
}

Image gray_world(const Image &im)
{
	Image result(im.n_rows, im.n_cols);
	double Sr = 0;
	double Sg = 0;
	double Sb = 0;
	for (int i = 0; i < im.n_rows; ++i)
		for (int j = 0; j < im.n_cols; ++j) {
			Sr += get<0>(im(i, j));
			Sg += get<1>(im(i, j));
			Sb += get<2>(im(i, j));
		}
	Sr /= im.n_rows * im.n_cols;
	Sg /= im.n_rows * im.n_cols;
	Sb /= im.n_rows * im.n_cols;
	double s = (Sr + Sg + Sb) / 3;
	for (int i = 0; i < im.n_rows; ++i)
		for (int j = 0; j < im.n_cols; ++j)
			result(i, j) = RGB(
				s * get<0>(im(i, j)) / Sr,
				s * get<1>(im(i, j)) / Sg,
				s * get<2>(im(i, j)) / Sb
			);
	return normalize(result);
}

Image unsharp(const Image &im)
{
	Matrix<double> kernel = {
		{-1.0/6, -2.0/3, -1.0/6},
		{-2.0/3, 13.0/3, -2.0/3},
		{-1.0/6, -2.0/3, -1.0/6}
	};
	return normalize(custom(im, kernel));
}

class ContrastFunctor
{
public:
	ContrastFunctor(int f)
		:threshold(f),
		 hyst(256, 0)
	{}
	
	RGB operator()(const Image &im)
	{
		int k = Y(
			get<0>(im(0,0)),
			get<1>(im(0,0)),
			get<2>(im(0,0))
		);
		hyst[k]++;
		return im(0, 0);
	}
	
	static Monochrome Y(Monochrome R, Monochrome G, Monochrome B)
	{
		return R * 0.2125 + G * 0.7154 + B * 0.0721;
	}

	pair<int, int> get_limits() const
	{
		int low_sum = 0;
		int i;
		for (i = 0; i < 256 && low_sum < threshold; ++i)
			low_sum += hyst[i];
		int j;
		int hi_sum = 0;
		for (j = 255; j >= 0 && hi_sum < threshold; --j)
			hi_sum += hyst[j];
		return make_pair(i, j);
	}
	
	double threshold;
	vector<int> hyst;
	static const int radius = 0;
};

class StretchHyst
{
public:
	StretchHyst(int t1, int t2)
		:threshold1(t1),
		 threshold2(t2)
	{}

	RGB operator()(const Image &im) const
	{
		return RGB(
			(get<0>(im(0, 0)) - threshold1) * 255 / (threshold2 - threshold1),
			(get<1>(im(0, 0)) - threshold1) * 255 / (threshold2 - threshold1),
			(get<2>(im(0, 0)) - threshold1) * 255 / (threshold2 - threshold1)
		);
	}
	int threshold1;
	int threshold2;
	static const int radius = 0;
};

Image autocontrast(const Image &im, double fraction)
{
	auto contrast = ContrastFunctor(fraction * im.n_rows * im.n_cols);
	auto result = im.unary_map(contrast);
	auto limits = contrast.get_limits();
	auto strech_hyst = StretchHyst(limits.first, limits.second);
	return normalize(result.unary_map(strech_hyst)); 
}

Image resize(const Image &im, double scale)
{
	Image out(im.n_rows * scale, im.n_cols * scale);
    for (int i = 0; i < out.n_rows; i++) {
            double tmp1 = (.0 + i) / (out.n_rows - 1) * (im.n_rows - 1);
            int h = floor(tmp1);
            h = max(0, min(im.n_rows - 2, h));
            double u = tmp1 - h;
        for (int j = 0; j < out.n_cols; j++) {
            double tmp2 = (.0 + j) / (out.n_cols - 1) * (im.n_cols - 1);
            int w = floor(tmp2);
            w = max(0, min(im.n_cols - 2, w));
            double t = tmp2 - w;
 
            double d1 = (1 - t) * (1 - u);
            double d2 = t * (1 - u);
            double d3 = t * u;
            double d4 = (1 - t) * u;
 
            RGB p1 = im(h, w);
            RGB p2 = im(h, w + 1);
            RGB p3 = im(h + 1, w + 1);
            RGB p4 = im(h + 1, w);
 			
 			out(i, j) = RGB(
	            get<0>(p1) * d1 +
	            get<0>(p2) * d2 +
	            get<0>(p3) * d3 +
	            get<0>(p4) * d4,

	            get<1>(p1) * d1 +
	            get<1>(p2) * d2 +
	            get<1>(p3) * d3 +
	            get<1>(p4) * d4,

	            get<2>(p1) * d1 +
	            get<2>(p2) * d2 +
	            get<2>(p3) * d3 +
	            get<2>(p4) * d4
			);
        }
    }
	return out;
}
