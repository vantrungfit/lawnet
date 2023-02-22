//
//  wristlib.h
//
//  Copyright © 2022 RB3043-NTUST. All rights reserved.
//

#ifndef WRISTLIB_H
#define WRISTLIB_H

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <boost/filesystem.hpp>
#include <boost/algorithm/algorithm.hpp>
#include <boost/format.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/opencv.hpp>
#include <corecrt_math_defines.h>

//constexpr auto M_PI = 3.141592653589793238463;

using namespace cv;
using namespace std;
namespace fs = boost::filesystem;

constexpr double M2PI = 2.0 * M_PI;
constexpr int binaryMaskSize = 160;
const int minRoiSize = saturate_cast<int>(0.25 * binaryMaskSize);
const Scalar greenColor = Scalar(0, 255, 0);
const Scalar whiteColor = Scalar(255, 255, 255);
const Scalar blackColor = Scalar(0, 0, 0);
const auto inputSize = cv::Size(640, 480);

inline double toDegree(double rad) {
    return rad * 180.0 / M_PI;
}

inline Point2i scalePoint(const Point2i& p, double scaleFactor) {
    return Point2i(saturate_cast<int>(p.x * scaleFactor), saturate_cast<int>(p.y * scaleFactor));
}

// Structs
typedef struct Landmarks {
    Point2i topLeft = Point2i(INT_MAX, INT_MAX);
    Point2i topRight = Point2i(INT_MAX, INT_MAX);
    Point2i bottomLeft = Point2i(INT_MAX, INT_MAX);
    Point2i bottomRight = Point2i(INT_MAX, INT_MAX);
    int edgeSize = INT_MAX;
    int angle = INT_MAX;

    void init(Point2i top_left, Point2i top_right, Point2i bottom_left, Point2i bottom_right) {
        topLeft = top_left;
        topRight = top_right;
        bottomLeft = bottom_left;
        bottomRight = bottom_right;
        const int dx = (topRight.x - topLeft.x);
        const int dy = (topRight.y - topLeft.y);
        edgeSize = saturate_cast<int>(sqrt(dx * dx + dy * dy));
        angle = saturate_cast<int>(toDegree(atan2(dy, dx)));
    };

    void translate(int biasX, int biasY) {
        topLeft = Point2i(topLeft.x + biasX, topLeft.y + biasY);
        topRight = Point2i(topRight.x + biasX, topRight.y + biasY);
        bottomLeft = Point2i(bottomLeft.x + biasX, bottomLeft.y + biasY);
        bottomRight = Point2i(bottomRight.x + biasX, bottomRight.y + biasY);
    }

    void resize(double scale) {
        topLeft = scalePoint(topLeft, scale);
        topRight = scalePoint(topRight, scale);
        bottomLeft = scalePoint(bottomLeft, scale);
        bottomRight = scalePoint(bottomRight, scale);
    }

    Point2f* inputQuad() {
        Point2f* input4 = new Point2f[4];

        input4[0] = Point2f(topLeft);
        input4[1] = Point2f(topRight);
        input4[2] = Point2f(bottomLeft);
        input4[3] = Point2f(bottomRight);

        return input4;
    };

    static Point2f* outputQuad(int roiSize = 128) {
        Point2f* output4 = new Point2f[4];
        output4[0] = Point2f(0.0f, 0.0f);
        output4[1] = Point2f(roiSize, 0);
        output4[2] = Point2f(0, roiSize);
        output4[3] = Point2f(roiSize, roiSize);

        return output4;
    };

} Landmarks;

typedef struct LineCoefficient {
    int a;
    int b;
    int c;
    int d;
    int d2;
} LineCoefficient;

// Inline Template Functions
template<typename T>
inline T clip(const T& a, const T& b, const T& c) {
    return min(max(a, b), c);
}

template<typename T>
inline int sign(const T& a) {
    return a < 0 ? -1 : a>0;
}

// Inline Functions

inline void normalizeIndex(int &index, const int max_val) {
    if(index<0) index += max_val;
    index %= max_val;
}

inline int cross(const Point2i& p1, const Point2i& p2) {
    return p1.x * p2.y - p1.y * p2.x;
}

inline int cross4(int a1, int b1, int a2, int b2) {
    return a1 * b2 - a2 * b1;
}

inline int magnitude(const Point2i& p) {
    return (int)(sqrt(p.x * p.x + p.y * p.y));
}

inline int magnitude2(int x, int y) {
    return x * x + y * y;
}

inline int midValue(int x, int y) {
    return (x + y) / 2;
}

inline Point2i midpoint(const Point2i& p1, const Point2i& p2) {
    return Point2i((p1.x + p2.x) / 2, (p1.y + p2.y) / 2);
}

inline Point2i sub(const Point2i& p1, const Point2i& p2) {
    return Point2i(p2.x - p1.x, p2.y - p1.y);
}

inline bool outBound(const Point2i& p, int xmin, int xmax, int ymin, int ymax) {
    return p.x<xmin || p.x>xmax || p.y<ymin || p.y>ymax;
}

inline int distance2Points(const Point2i& p1, const Point2i& p2) {
    const Point2i p = sub(p1, p2);
    return magnitude(p);
}

inline double angle3pt(
    const Point2i& p1,
    const Point2i& p2,
    const Point2i& p3) {
    double a = atan2(p3.y - p2.y, p3.x - p2.x) - atan2(p1.y - p2.y, p1.x - p2.x);

    if (a < 0) a += M2PI;

    if (a > M_PI) a = M2PI - a;

    return toDegree(a);
}

inline double angle2VectWithDirection(
    const Point2i& p1,
    const Point2i& p2
) {

    double a = atan2(-p2.y, p2.x) - atan2(-p1.y, p1.x);

    if (a > M_PI) a -= M2PI;
    if (a < -M_PI) a += M2PI;

    return toDegree(a);
}

inline LineCoefficient getLineCoefficient(
    const Point2i& p1,
    const Point2i& p2
) {
    const auto p = sub(p1, p2);
    LineCoefficient coef;
    coef.a = p.y;
    coef.b = -p.x;
    coef.c = cross(p2, p1);
    coef.d2 = magnitude2(p.x, p.y);
    coef.d = saturate_cast<int>(sqrt(coef.d2));

    return coef;
}

inline int pointLineScore(
    const Point2i& p,
    const LineCoefficient& coef
) {
    return coef.a * p.x + coef.b * p.y + coef.c;
}

inline int pointLineDistance(
    const Point2i& p,
    const LineCoefficient& coef) {

    if (coef.d == 0) return INT_MAX;

    return abs(pointLineScore(p, coef)) / coef.d;
}


void AGC(Mat& src, double gamma, bool isAutoMode = true);

void enhanceVein(Mat& src);

bool extractWristVein(
    const Mat& rgbImage,
    Mat& veinImage,
    Mat& binaryImage,
    Mat& roiImage,
    Landmarks& landmarks,
    int code2RGB,
    int roiSize = 128
);

#endif // WRISTLIB_H