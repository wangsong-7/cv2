/**

 *  功能菜单：
 *   0. 灰度化          (Grayscale)
 *   1. 高斯滤波        (Gaussian Blur)
 *   2. 中值滤波        (Median Blur)
 *   3. 二值化          (Threshold)
 *   4. Canny 边缘检测  (Canny)
 *   5. 形态学操作      (Morphology)
 *   6. 轮廓提取        (Contour)
 *   7. 全流水线        (Full Pipeline)
 *   q. 退出
 * ============================================================
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <map>
#include <functional>
#include <windows.h>
using namespace cv;
using namespace std;

// ────────────────────────────────────────────────────────────
//  工具函数
// ────────────────────────────────────────────────────────────

/** 安全读取图片，失败时抛出异常 */
Mat loadImage(const string& path)
{
    Mat img = imread(path, IMREAD_COLOR);
    if (img.empty()) {
        throw runtime_error("无法读取图片: " + path);
    }
    cout << "[✓] 已加载: " << path
         << "  尺寸: " << img.cols << "x" << img.rows
         << "  通道: " << img.channels() << endl;
    return img;
}

/** 显示并保存结果 */
void showAndSave(const string& winName, const Mat& img, const string& savePath)
{
    imshow(winName, img);
    imwrite(savePath, img);
    cout << "[✓] 结果已保存: " << savePath << endl;
    cout << "    按任意键继续..." << endl;
    waitKey(0);
    destroyWindow(winName);
}

// ────────────────────────────────────────────────────────────
//  各处理模块
// ────────────────────────────────────────────────────────────

/**
 * 0. 灰度化
 *    COLOR_BGR2GRAY：将 3 通道 BGR 转为单通道灰度图
 */
Mat processGrayscale(const Mat& src)
{
    Mat gray;
    cvtColor(src, gray, COLOR_BGR2GRAY);
    return gray;
}

/**
 * 1. 高斯滤波
 *    GaussianBlur：用高斯核对图像做卷积，平滑噪声
 *    ksize：核大小（必须为奇数），sigmaX：X 方向标准差
 */
Mat processGaussianBlur(const Mat& src, int ksize = 5, double sigma = 0)
{
    // ksize 必须是奇数
    if (ksize % 2 == 0) ksize += 1;
    Mat blurred;
    GaussianBlur(src, blurred, Size(ksize, ksize), sigma);
    return blurred;
}

/**
 * 2. 中值滤波
 *    medianBlur：用邻域像素的中值替换中心像素，对椒盐噪声效果好
 *    ksize：滤波窗口大小（奇数）
 */
Mat processMedianBlur(const Mat& src, int ksize = 5)
{
    if (ksize % 2 == 0) ksize += 1;
    Mat blurred;
    medianBlur(src, blurred, ksize);
    return blurred;
}

/**
 * 3. 二值化
 *    threshold：将灰度图按阈值转为黑白二值图
 *    THRESH_BINARY：大于阈值→maxval，否则→0
 *    THRESH_OTSU：自动计算最优阈值（传入 thresh=0 即可）
 */
Mat processThreshold(const Mat& src, double thresh = 0, double maxval = 255)
{
    Mat gray = (src.channels() == 1) ? src : processGrayscale(src);
    Mat binary;
    // THRESH_OTSU 会自动计算最优阈值
    double usedThresh = threshold(gray, binary, thresh, maxval,
                                  THRESH_BINARY | THRESH_OTSU);
    cout << "    [Otsu 自动阈值] = " << usedThresh << endl;
    return binary;
}

/**
 * 4. Canny 边缘检测
 *    步骤：灰度 → 高斯平滑 → 梯度计算 → 非极大值抑制 → 双阈值
 *    low/high：双阈值，建议比例约 1:2 或 1:3
 */
Mat processCanny(const Mat& src, double low = 50, double high = 150)
{
    Mat gray = (src.channels() == 1) ? src : processGrayscale(src);
    Mat blurred;
    GaussianBlur(gray, blurred, Size(5, 5), 0);
    Mat edges;
    Canny(blurred, edges, low, high);
    return edges;
}

/**
 * 5. 形态学操作
 *    getStructuringElement：创建结构元素（核）
 *    morphologyEx：执行形态学变换
 *      MORPH_ERODE    腐蚀：缩小白色区域
 *      MORPH_DILATE   膨胀：扩大白色区域
 *      MORPH_OPEN     开运算（先腐蚀后膨胀）：去除小噪点
 *      MORPH_CLOSE    闭运算（先膨胀后腐蚀）：填充小孔洞
 *      MORPH_GRADIENT 形态学梯度：轮廓提取
 */
Mat processMorphology(const Mat& src, int op = MORPH_CLOSE, int ksize = 5)
{
    Mat binary = processThreshold(src);
    Mat kernel = getStructuringElement(MORPH_RECT, Size(ksize, ksize));
    Mat result;
    morphologyEx(binary, result, op, kernel);

    map<int, string> opName = {
        {MORPH_ERODE,    "腐蚀 (Erode)"},
        {MORPH_DILATE,   "膨胀 (Dilate)"},
        {MORPH_OPEN,     "开运算 (Open)"},
        {MORPH_CLOSE,    "闭运算 (Close)"},
        {MORPH_GRADIENT, "形态学梯度 (Gradient)"},
    };
    cout << "    [形态学操作] = " << opName[op] << endl;
    return result;
}

/**
 * 6. 轮廓提取
 *    findContours：从二值图中查找所有轮廓
 *    drawContours：将轮廓绘制到彩色图上
 *    RETR_EXTERNAL：只取最外层轮廓
 *    CHAIN_APPROX_SIMPLE：压缩水平/垂直/对角线段，只保留端点
 */
Mat processContour(const Mat& src)
{
    Mat binary = processThreshold(src);

    // 形态学开运算去噪，使轮廓更干净
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(binary, binary, MORPH_OPEN, kernel);

    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(binary, contours, hierarchy,
                 RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // 在原图彩色副本上绘制轮廓
    Mat result;
    if (src.channels() == 1)
        cvtColor(src, result, COLOR_GRAY2BGR);
    else
        result = src.clone();

    // 过滤面积过小的轮廓（去除噪点）
    int validCount = 0;
    for (size_t i = 0; i < contours.size(); ++i) {
        if (contourArea(contours[i]) > 50) {
            drawContours(result, contours, (int)i,
                         Scalar(0, 255, 0), 2);
            ++validCount;
        }
    }
    cout << "    [轮廓数量] 共检测到 " << validCount << " 个有效轮廓" << endl;
    return result;
}

/**
 * 7. 全流水线
 *    依次执行：灰度 → 高斯滤波 → 二值化 → 形态学 → 轮廓提取
 *    每步结果拼接为对比图输出
 */
Mat processFullPipeline(const Mat& src)
{
    cout << "    [全流水线] 依次执行各步骤..." << endl;

    Mat gray    = processGrayscale(src);
    Mat blurred = processGaussianBlur(gray, 5);
    Mat binary  = processThreshold(blurred);
    Mat morph   = processMorphology(binary, MORPH_CLOSE, 5);
    Mat contour = processContour(morph);

    // ── 拼图：将 5 步结果横向拼接 ──
    // 统一转 BGR 方便拼接
    auto toBGR = [](const Mat& m) -> Mat {
        Mat out;
        if (m.channels() == 1) cvtColor(m, out, COLOR_GRAY2BGR);
        else out = m.clone();
        return out;
    };

    // 统一缩放到同一高度
    int targetH = 300;
    auto resize_h = [&](const Mat& m) -> Mat {
        double scale = (double)targetH / m.rows;
        Mat out;
        resize(m, out, Size(), scale, scale, INTER_LINEAR);
        return out;
    };

    vector<Mat> panels = {
        resize_h(toBGR(src)),
        resize_h(toBGR(gray)),
        resize_h(toBGR(blurred)),
        resize_h(toBGR(binary)),
        resize_h(toBGR(morph)),
        resize_h(toBGR(contour))
    };

    // 加标签
    vector<string> labels = {
        "Original", "Gray", "Gaussian", "Threshold", "Morphology", "Contour"
    };
    for (size_t i = 0; i < panels.size(); ++i) {
        putText(panels[i], labels[i], Point(8, 26),
                FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 200, 255), 2);
    }

    Mat row;
    hconcat(panels, row);
    return row;
}

// ────────────────────────────────────────────────────────────
//  主菜单
// ────────────────────────────────────────────────────────────

void printMenu()
{
    cout << "\n╔══════════════════════════════════════╗\n";
    cout << "║     image_preprocess  工具菜单       ║\n";
    cout << "╠══════════════════════════════════════╣\n";
    cout << "║  0. 灰度化        (Grayscale)        ║\n";
    cout << "║  1. 高斯滤波      (Gaussian Blur)    ║\n";
    cout << "║  2. 中值滤波      (Median Blur)      ║\n";
    cout << "║  3. 二值化        (Threshold/Otsu)   ║\n";
    cout << "║  4. Canny 边缘    (Canny)            ║\n";
    cout << "║  5. 形态学操作    (Morphology)       ║\n";
    cout << "║  6. 轮廓提取      (Contour)          ║\n";
    cout << "║  7. 全流水线对比  (Full Pipeline)    ║\n";
    cout << "║  q. 退出                             ║\n";
    cout << "╚══════════════════════════════════════╝\n";
    cout << "请选择 [0-7 / q]: ";
}

int main(int argc, char* argv[])
{
    // ── 获取图片路径 ──
    SetConsoleOutputCP(65001);
    SetConsoleCP(65001);
    string imagePath;
    if (argc >= 2) {
        imagePath = argv[1];
    } else {
        cout << "请输入图片路径（例如 D:/images/test.jpg）: ";
        cin >> imagePath;
    }

    Mat src;
    try {
        src = loadImage(imagePath);
    } catch (const exception& e) {
        cerr << "[✗] " << e.what() << endl;
        return -1;
    }

    // ── 交互循环 ──
    char choice;
    while (true) {
        printMenu();
        cin >> choice;

        if (choice == 'q' || choice == 'Q') {
            cout << "再见！" << endl;
            break;
        }

        Mat result;
        string suffix;
        string winName;

        try {
            switch (choice) {
            case '0':
                result  = processGrayscale(src);
                suffix  = "_gray.jpg";
                winName = "Grayscale";
                break;
            case '1':
                result  = processGaussianBlur(src, 9);
                suffix  = "_gaussian.jpg";
                winName = "Gaussian Blur";
                break;
            case '2':
                result  = processMedianBlur(src, 7);
                suffix  = "_median.jpg";
                winName = "Median Blur";
                break;
            case '3':
                result  = processThreshold(src);
                suffix  = "_thresh.jpg";
                winName = "Threshold (Otsu)";
                break;
            case '4':
                result  = processCanny(src, 50, 150);
                suffix  = "_canny.jpg";
                winName = "Canny Edges";
                break;
            case '5': {
                // 让用户选择形态学操作类型
                cout << "  形态学操作类型：\n"
                     << "    0: 腐蚀   1: 膨胀   2: 开运算\n"
                     << "    3: 闭运算 4: 梯度\n"
                     << "  请选择 [0-4]: ";
                int op; cin >> op;
                int morphOps[] = {MORPH_ERODE, MORPH_DILATE,
                                  MORPH_OPEN,  MORPH_CLOSE, MORPH_GRADIENT};
                if (op < 0 || op > 4) op = 3;
                result  = processMorphology(src, morphOps[op], 5);
                suffix  = "_morphology.jpg";
                winName = "Morphology";
                break;
            }
            case '6':
                result  = processContour(src);
                suffix  = "_contour.jpg";
                winName = "Contours";
                break;
            case '7':
                result  = processFullPipeline(src);
                suffix  = "_pipeline.jpg";
                winName = "Full Pipeline";
                break;
            default:
                cout << "[!] 无效选项，请重新输入。" << endl;
                continue;
            }

            // 生成输出路径：在原文件名基础上加后缀
            string outPath = imagePath.substr(0, imagePath.find_last_of('.')) + suffix;
            showAndSave(winName, result, outPath);

        } catch (const exception& e) {
            cerr << "[✗] 处理出错: " << e.what() << endl;
        }
    }

    return 0;
}