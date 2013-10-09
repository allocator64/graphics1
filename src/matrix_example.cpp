#include <iostream>
#include "matrix.h"
#include "io.h"

using std::cout;
using std::endl;

using std::tuple;
using std::get;
using std::tie;
using std::make_tuple;

// Matrix usage example
// Also see: matrix.h, matrix.hpp for comments on how filtering works

class BoxFilterOp
{
public:
    tuple<int, int, int> operator () (const Image &m) const
    {
        int size = 2 * radius + 1;
        int r, g, b, sum_r = 0, sum_g = 0, sum_b = 0;
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                // Tie is useful for taking elements from tuple
                tie(r, g, b) = m(i, j);
                sum_r += r;
                sum_g += g;
                sum_b += b;
            }
        }
        auto norm = size * size;
        sum_r /= norm;
        sum_g /= norm;
        sum_b /= norm;
        return make_tuple(sum_r, sum_g, sum_b);
    }
    // Radius of neighbourhoud, which is passed to that operator
    static const int radius = 1;
};

int main(int argc, char **argv)
{
    // Image = Matrix<tuple<int, int, int>>
    // tuple is from c++ 11 standard
    Image img = load_image(argv[1]);
    Image img3;
    img3 = img;
    Image img2 = img3.unary_map(BoxFilterOp());
    save_image(img2, argv[2]);
}
