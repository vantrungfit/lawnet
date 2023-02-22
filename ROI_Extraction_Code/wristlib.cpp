#include "wristlib.h"

int biasX = 0, biasY = 0;
double scaleFactor = 2.0;
const double gammaValue = 1.0;
const double roiMargin = 0.1;
const double laplacianDelta = 0.0;
const double guideCircleRadiusFactor = 0.3;
const int lineThickness = 2;
const int circleRadian = 6;
const auto lineColor = Scalar(120, 120, 120);
const auto circleColor = Scalar(100, 100, 100);
const double whiteArea_threshold = 0.75;
const double roiShape_threshold = 0.7;

void AGC(Mat& src, double gamma, bool isAutoMode) {
    double c = 1.0;
    double d = 1.0 / 255.0;

    if (isAutoMode) {
        double val = mean(src).val[0] / 255;
        c = log10(0.5) / log10(val);
    }
    gamma *= c;
    Mat lut(1, 256, CV_8U);

    for (int i = 0; i < 256; ++i) {
        lut.at<uchar>(i) = saturate_cast<uchar>(clip(pow(i * d, gamma) * 255.0, 0.0, 255.0));
    }
    LUT(src, lut, src);
}

void enhanceVein(Mat& src) {

    if (src.data == nullptr) return;

    // Automatic Gammar Correction
    AGC(src, gammaValue, true);

    //Sharpen Image using CLAHE
    auto clahe = createCLAHE();
    clahe->setClipLimit(2);
    clahe->setTilesGridSize(cv::Size(4, 4));
    clahe->apply(src, src);

    //Low-pass Gaussian filter
    src.convertTo(src, CV_64F, 1.0 / 255.0);
    GaussianBlur(src, src, cv::Size(0, 0), 3);

    //High-pass Laplacian filter
    Laplacian(src, src, CV_64F, 1, 1, laplacianDelta);
    src = max(src, 0.0);
    double lapmin, lapmax;
    minMaxLoc(src, &lapmin, &lapmax);
    const double scale = 255.0 / max(-lapmin, lapmax);
    src *= scale;
    src.convertTo(src, CV_8UC1);
}

bool extractWristVein(
    const Mat& wristImage,
    Mat& veinImage,
    Mat& binaryImage,
    Mat& roiImage,
    Landmarks& landmarks,
    int code2RGB,
    int roiSize
) {
    Mat tmp, rgbImage;
    cvtColor(wristImage, rgbImage, code2RGB);

    cvtColor(rgbImage, tmp, COLOR_RGB2HLS);
    extractChannel(tmp, veinImage, 2);

    Mat image;
    rgbImage.copyTo(image);

    // 1. SCALE DOWN
    scaleFactor = image.rows * 1.0 / binaryMaskSize;

    if (image.rows > binaryMaskSize) {
        resize(image, image, cv::Size(binaryMaskSize, binaryMaskSize), 0, 0, INTER_AREA);
    }
    else if (image.rows < binaryMaskSize) {
        resize(image, image, cv::Size(binaryMaskSize, binaryMaskSize), 0, 0, INTER_LINEAR);
    }

    // 2. SEGMENT WRIST REGION
    vector<Mat> lab, yrb;
    vector<vector<Point2i> > contours;
    double new_area;
    double largest_area;
    size_t largest_contour_index;

    // Find most compact channel between channels A* and B* of LAB color space, Cr and Cb of YCrCb color space
    cvtColor(image, tmp, COLOR_RGB2Lab);
    split(tmp, lab);

    cvtColor(image, tmp, COLOR_RGB2YCrCb);
    split(tmp, yrb);

    auto bin_size = binaryMaskSize * binaryMaskSize;
    const auto gaussKernel = cv::Size(9, 9);
    Mat bin_a, bin_b, bin_cr, bin_cb;

    GaussianBlur(lab.at(1), lab.at(1), gaussKernel, 0);
    threshold(lab.at(1), bin_a, 0, 255, THRESH_BINARY + THRESH_TRIANGLE);
    auto score_a = countNonZero(bin_a)*1.0/bin_size;
    
    GaussianBlur(lab.at(2), lab.at(2), gaussKernel, 0);
    threshold(lab.at(2), bin_b, 0, 255, THRESH_BINARY + THRESH_TRIANGLE);
    auto score_b = countNonZero(bin_b)*1.0 / bin_size;
   
    GaussianBlur(yrb.at(1), yrb.at(1), gaussKernel, 0);
    threshold(yrb.at(1), bin_cr, 0, 255, THRESH_BINARY + THRESH_TRIANGLE);
    auto score_cr = countNonZero(bin_cr) * 1.0 / bin_size;

    GaussianBlur(yrb.at(2), yrb.at(2), gaussKernel, 0);
    threshold(yrb.at(2), bin_cb, 0, 255, THRESH_BINARY_INV + THRESH_TRIANGLE);
    auto score_cb = countNonZero(bin_cb) * 1.0 / bin_size;

    double best_score = 0;

    if (score_a<whiteArea_threshold && score_a>best_score) {
        best_score = score_a;
    }

    if (score_b<whiteArea_threshold && score_b>best_score) {
        best_score = score_b;
    }

    if (score_cr<whiteArea_threshold && score_cr>best_score) {
        best_score = score_cr;
    }

    if (score_cb<whiteArea_threshold && score_cb>best_score) {
        best_score = score_cb;
    }

    if (best_score == score_a) {
        binaryImage = bin_a;
    }
    else if (best_score == score_b) {
        binaryImage = bin_b;
    }
    else if (best_score == score_cr) {
        binaryImage = bin_cr;
    }
    else {
        binaryImage = bin_cb;
    }
    
    // 3. FIND THE LARGEST CONTOUR
    largest_area = 0.0;
    largest_contour_index = 0;
    findContours(binaryImage, contours, RETR_LIST, CHAIN_APPROX_NONE);
    const auto contours_size = contours.size();

    for (size_t i = 0; i < contours_size; ++i) {
        new_area = contourArea(contours.at(i));

        if (new_area > largest_area) {
            largest_area = new_area;
            largest_contour_index = i;
        }
    }
    auto contour = contours.at(largest_contour_index);
    const auto num_contour_points = contour.size();
    binaryImage.setTo(0);
    drawContours(binaryImage, contours, largest_contour_index, Scalar(255.0), FILLED);

    // 4. FIND CONVEX HULL AND CONVEXITY DEFECTS
    vector<int> cvHull;
    convexHull(contour, cvHull, false, false);
    std::vector<Vec4i> defects;
    convexityDefects(contour, cvHull, defects);

    auto defect = *max_element(defects.begin(), defects.end(),
        [](const auto& a, const auto& b) {
            return a[3] < b[3];
        });

    // 5. DETERMINE THE KEY VECTOR
    auto p1 = contour.at(defect[0]);
    auto p2 = contour.at(defect[1]);
    const auto p3 = contour.at(defect[2]);
    auto step1 = sign(defect[0] - defect[1]);
    auto step2 = -step1;

    // swap p1 and p2 if distance from p1 to p3 is greater than distance from p2 to p3
    const auto d1 = distance2Points(p3, p1);
    const auto d2 = distance2Points(p3, p2);

    if (d1 > d2) {
        auto t = defect[0];
        defect[0] = defect[1];
        defect[1] = t;

        p1 = contour.at(defect[0]);
        p2 = contour.at(defect[1]);

        t = step1;
        step1 = step2;
        step2 = t;
    }
    

    // 6. LOCATE WRIST ROI
    // line passes through the farthest point p3 and 
    // perpendicular to p2p3 and intersects with contour at p4
    auto coef = getLineCoefficient(p2, p3);
    const auto t = coef.a;
    coef.a = -coef.b;
    coef.b = t;
    coef.c = -(coef.a * p3.x + coef.b * p3.y);

    auto k4 = defect[0] + step1;
    normalizeIndex(k4, num_contour_points);
    auto p4 = contour.at(k4);
    auto sign1 = sign(pointLineDistance(p4, coef));
    int sign2 = sign1;

    while (sign2 == sign1) {
        k4 += step1;
        normalizeIndex(k4, num_contour_points);
        p4 = contour.at(k4);
        sign2 = sign(pointLineDistance(p4, coef));
    }

    if (sign2 != 0) {
        k4 -= step1;
        normalizeIndex(k4, num_contour_points);
        p4 = contour.at(k4);
    }

    // find wrist scale p5p6
    // scale line is parallel to line p3p4 and passes through midpoint p5 of p2 and p3
    const auto k5 = midValue(defect[1], defect[2]);
    const auto p5 = contour.at(k5);
    auto k6 = k4+step1;
    normalizeIndex(k6, num_contour_points);
    auto p6 = contour.at(k6);
    coef.c = -(coef.a * p5.x + coef.b * p5.y);
    sign1 = sign(pointLineDistance(p4, coef));
    sign2 = sign1;

    while (sign2 == sign1) {
        k6 += step1;
        normalizeIndex(k6, num_contour_points);
        p6 = contour.at(k6);
        sign2 = sign(pointLineDistance(p6, coef));
    }

    if (sign2 != 0) {
        k6 -= step1;
        normalizeIndex(k6, num_contour_points);
        p6 = contour.at(k6);
    }

    const auto top_edge = distance2Points(p3, p4);
    const auto scale = distance2Points(p5, p6);

    if (scale < minRoiSize) {
        cout << "ROI scale is too small!" << endl;
        return false;
    }

    if (scale < roiShape_threshold * top_edge) {
        cout << "ROI scale is too small compared with top edge!" << endl;
        return false;
    }

    // Find midpoint of bottom edge (mid_bottom)
    // Distance from midpoint of top edge p3p4 (mid_top) to mid_bottom is equal to scale p5p6
    const auto mid_top = midpoint(p3, p4);
    const auto mid56 = midpoint(p5, p6);
    const auto coef1 = getLineCoefficient(mid_top, mid56);
    auto m = 1.0;
    auto k = (1.0-roiMargin)*scale;

    if (coef1.b != 0) {
        m = abs(coef1.a * 1.0 / coef1.b);
        k *= sqrt(1.0 / (1.0 + m * m));
    }

    auto mid_bottom = sub(mid_top, mid56);
    mid_bottom.x = (int)(mid_top.x + sign(mid_bottom.x) * k);
    mid_bottom.y = (int)(mid_top.y + sign(mid_bottom.y) * m * k);

    if (outBound(mid_bottom, 0, binaryImage.cols, 0, binaryImage.rows)) {
        cout << "Outbound" << endl;
        return false;
    }

    line(binaryImage, p3, p4, lineColor, lineThickness);
    line(binaryImage, p5, p6, lineColor, lineThickness);
    line(binaryImage, mid_top, mid_bottom, lineColor, lineThickness);

    circle(binaryImage, mid_top, circleRadian, circleColor, lineThickness);
    circle(binaryImage, mid56, circleRadian, circleColor, lineThickness);
    circle(binaryImage, mid_bottom, circleRadian, circleColor, lineThickness);
    circle(binaryImage, p1, circleRadian, circleColor, lineThickness);
    circle(binaryImage, p2, circleRadian, circleColor, lineThickness);
    circle(binaryImage, p3, circleRadian, circleColor, lineThickness);
    circle(binaryImage, p4, circleRadian, circleColor, lineThickness);
    circle(binaryImage, p5, circleRadian, circleColor, lineThickness);
    circle(binaryImage, p6, circleRadian, circleColor, lineThickness);


    // Find bottom left and bottom right on contour
    // lay on line which is parallel to top edge and passes through mid_bottom
    coef.c = -(coef.a * mid_bottom.x + coef.b * mid_bottom.y);
    // bottom left
    auto k_left = defect[2] + step2;
    normalizeIndex(k_left, num_contour_points);
    landmarks.bottomLeft = contour.at(k_left);
    sign1 = sign(pointLineDistance(landmarks.bottomLeft, coef));
    sign2 = sign1;

    while (sign2 == sign1 && k_left != k4) {
        k_left += step2;
        normalizeIndex(k_left, num_contour_points);
        landmarks.bottomLeft = contour.at(k_left);
        sign2 = sign(pointLineDistance(landmarks.bottomLeft, coef));
    }

    if (sign2 != 0 && sign2 != sign1) {
        k_left -= step2;
        normalizeIndex(k_left, num_contour_points);
        landmarks.bottomLeft = contour.at(k_left);
    }
    else if (sign2 == sign1) {
        cout << "Bottom left not found!" << endl;
        return false;
    }

    // bottom right
    auto k_right = k4 + step1;
    normalizeIndex(k_right, num_contour_points);
    landmarks.bottomRight = contour.at(k_right);
    sign1 = sign(pointLineDistance(landmarks.bottomRight, coef));
    sign2 = sign1;

    while (sign2 == sign1 && k_right != k_left) {
        k_right += step1;
        normalizeIndex(k_right, num_contour_points);
        landmarks.bottomRight = contour.at(k_right);
        sign2 = sign(pointLineDistance(landmarks.bottomRight, coef));
    }

    if (sign2 != 0 && sign2 != sign1) {
        k_right -= step1;
        normalizeIndex(k_right, num_contour_points);
        landmarks.bottomRight = contour.at(k_right);
    }
    else if (sign2 == sign1) {
        cout << "Bottom right not found!" << endl;
        return false;
    }

    line(binaryImage, landmarks.bottomLeft, landmarks.bottomRight, lineColor, lineThickness);
    circle(binaryImage, landmarks.bottomLeft, circleRadian, circleColor, lineThickness);
    circle(binaryImage, landmarks.bottomRight, circleRadian, circleColor, lineThickness);

    const auto bottom_edge = distance2Points(landmarks.bottomLeft, landmarks.bottomRight);

    if (bottom_edge < roiShape_threshold * scale) {
        cout << "Bottom edge is too small compared with scale!" << endl;
        return false;
    }

    // specify top_left, top_right, bottom_left, bottom_right
    const auto angle = angle2VectWithDirection(sub(p3, p1), sub(p3, p4));
    landmarks.topLeft = p3;
    landmarks.topRight = p4;

    if (angle > 0) {
        landmarks.topLeft = p4;
        landmarks.topRight = p3;
        const auto pt = landmarks.bottomLeft;
        landmarks.bottomLeft = landmarks.bottomRight;
        landmarks.bottomRight = pt;
    }
    
    // adjust ROI Top left and Top right..." << endl;
    auto margin = roiMargin * top_edge;
    m = 1.0;
    k = margin;

    if (coef.b != 0) {
        m = abs(coef.a * 1.0 / coef.b);
        k *= sqrt(1.0 / (1.0 + m * m));
    }
    // adjust top left
    auto delta = sub(landmarks.topLeft, landmarks.topRight);
    landmarks.topLeft.x = (int)(landmarks.topLeft.x + sign(delta.x) * k);
    landmarks.topLeft.y = (int)(landmarks.topLeft.y + sign(delta.y) * m * k);
    // adjust top right
    delta = sub(landmarks.topRight, landmarks.topLeft);
    landmarks.topRight.x = (int)(landmarks.topRight.x + sign(delta.x) * k);
    landmarks.topRight.y = (int)(landmarks.topRight.y + sign(delta.y) * m * k);
    
    // adjust ROI Bottom left and Bottom right...";
    margin = roiMargin * bottom_edge;
    k = margin;

    if (coef.b != 0) {
        k *= sqrt(1.0 / (1.0 + m * m));
    }
    // adjust bottom left
    delta = sub(landmarks.bottomLeft, landmarks.bottomRight);
    landmarks.bottomLeft.x = (int)(landmarks.bottomLeft.x + sign(delta.x) * k);
    landmarks.bottomLeft.y = (int)(landmarks.bottomLeft.y + sign(delta.y) * m * k);
    // adjust bottom right
    delta = sub(landmarks.bottomRight, landmarks.bottomLeft);
    landmarks.bottomRight.x = (int)(landmarks.bottomRight.x + sign(delta.x) * k);
    landmarks.bottomRight.y = (int)(landmarks.bottomRight.y + sign(delta.y) * m * k);
    
    line(binaryImage, landmarks.topLeft, landmarks.bottomLeft, lineColor, lineThickness);
    line(binaryImage, landmarks.topRight, landmarks.bottomRight, lineColor, lineThickness);

    circle(binaryImage, landmarks.topLeft, circleRadian, circleColor, lineThickness);
    circle(binaryImage, landmarks.topRight, circleRadian, circleColor, lineThickness);
    circle(binaryImage, landmarks.bottomLeft, circleRadian, circleColor, lineThickness);
    circle(binaryImage, landmarks.bottomRight, circleRadian, circleColor, lineThickness);

    // scale back to image size
    landmarks.resize(scaleFactor);

    // 7. EXTRACT WRIST VEIN ROI IMAGE
    const auto dsize = cv::Size(roiSize, roiSize);
    const auto input4 = landmarks.inputQuad();
    auto output4 = Landmarks::outputQuad(roiSize);
    auto M = getPerspectiveTransform(input4, output4);

    roiImage = Mat(roiSize, roiSize, CV_8UC1);
    warpPerspective(veinImage, roiImage, M, dsize);

    // enhance wrist vein
    enhanceVein(roiImage);
    
    landmarks.translate(biasX, biasY);
    
    return true;
}

bool extractWristVein(Mat& frame, Mat& binaryImage, Mat& roiImage, Mat& veinImage, int preferenceCode, int frame2RGBCode, int gray2FrameCode, int roiSize) {
    
    Mat wristImage, rgbImage, tmp;
    Landmarks landmarks;

    // 1. CROP TO SQUARE
    cv::Rect rect1, rect2, rect3;
    const bool portrait = frame.cols < frame.rows;

    if (portrait) {
        biasY = (frame.rows - frame.cols) / 2;
        rect1 = cv::Rect(0, 0, frame.cols, biasY);
        rect2 = cv::Rect(0, biasY, frame.cols, frame.cols);
        rect3 = cv::Rect(0, biasY + frame.cols, frame.cols, biasY);
    }
    else {
        biasX = (frame.cols - frame.rows) / 2;
        rect1 = cv::Rect(0, 0, biasX, frame.rows);
        rect2 = cv::Rect(biasX, 0, frame.rows, frame.rows);
        rect3 = cv::Rect(biasX + frame.rows, 0, biasX, frame.rows);
    }
    frame(rect2).copyTo(wristImage);

    // Wrist Vein ROI Extraction
    const bool ok = extractWristVein(wristImage, veinImage, binaryImage, roiImage, landmarks, frame2RGBCode, roiSize);
    const int imageFormat = preferenceCode / 10;

    if (imageFormat == 2) {
        cvtColor(veinImage, tmp, gray2FrameCode);
        tmp.copyTo(frame(rect2));
    }
    else if (imageFormat == 3) {
        resize(binaryImage, tmp, cv::Size(veinImage.cols, veinImage.rows));
        cvtColor(tmp, tmp, gray2FrameCode);
        tmp.copyTo(frame(rect2));
    }
    else if (imageFormat == 4) {
        if (!roiImage.empty()) {
            resize(roiImage, tmp, cv::Size(veinImage.cols, veinImage.rows));
        }
        cvtColor(tmp, tmp, gray2FrameCode);
        tmp.copyTo(frame(rect2));
    }
    frame(rect1).setTo(blackColor);
    frame(rect3).setTo(blackColor);

    // Draw guide line
    if (!ok) {
        circle(frame, cv::Point(frame.cols / 2.0, frame.rows / 2.0),
            (int)(min(frame.cols, frame.rows) * guideCircleRadiusFactor), greenColor, lineThickness);

        return false;
    }

    // Draw bounding box
    if (imageFormat != 4) {

        line(frame, landmarks.topLeft, landmarks.topRight, greenColor, lineThickness);
        line(frame, landmarks.topRight, landmarks.bottomRight, greenColor, lineThickness);
        line(frame, landmarks.bottomRight, landmarks.bottomLeft, greenColor, lineThickness);
        line(frame, landmarks.bottomLeft, landmarks.topLeft, greenColor, lineThickness);
    }

    return true;
}

int main(const int argc, char** argv)
{
    if (argc == 1) return 1;

    const int preferenceCode = 21;
    const auto roiSize = 128;
    const auto frame2RGBCode = COLOR_BGR2RGB;
    const auto gray2FrameCode = COLOR_GRAY2BGR;
    const string input_path = argv[1];

    Mat binaryImage, roiImage, veinImage;
    Mat frame = imread(input_path, IMREAD_UNCHANGED);

    extractWristVein(frame, binaryImage, roiImage, veinImage, preferenceCode, frame2RGBCode, gray2FrameCode, roiSize);

    if (frame.data != nullptr) imwrite("camera_frame.png", frame);

    if (binaryImage.data != nullptr) imwrite("binary_image.png", roiImage);

    if (roiImage.data != nullptr) imwrite("roi_image.png", roiImage);

    if (veinImage.data != nullptr) imwrite("vein_image.png", roiImage);

    return 0;
}