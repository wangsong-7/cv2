/*
 * task2.cpp
 * 功能：视觉测量 —— 圆形目标点识别与物理坐标提取
 * 对应MATLAB脚本 s25.m
 * 依赖：OpenCV 4.x
 *
 * 编译示例（MinGW/MSVC均适用，请修改OpenCV路径）：
 *   g++ task2.cpp -o task2.exe -I<opencv_include> -L<opencv_lib> -lopencv_world4xx
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>

// ============================================================
//  子函数：最小二乘椭圆拟合 (SVD)
//  输入：边缘点坐标 X, Y（局部坐标）
//  输出：params = [A B C D E F]，满足 Ax²+Bxy+Cy²+Dx+Ey+F=0
//        is_valid：B²-4AC<0 则为真椭圆
// ============================================================
bool fitEllipseLS(const std::vector<cv::Point>& pts, std::vector<double>& params)
{
    int n = (int)pts.size();
    if (n < 6) return false;

    // 构建设计矩阵 M (n×6)
    cv::Mat M(n, 6, CV_64F);
    for (int i = 0; i < n; i++) {
        double x = pts[i].x;
        double y = pts[i].y;
        M.at<double>(i, 0) = x * x;
        M.at<double>(i, 1) = x * y;
        M.at<double>(i, 2) = y * y;
        M.at<double>(i, 3) = x;
        M.at<double>(i, 4) = y;
        M.at<double>(i, 5) = 1.0;
    }

    // SVD 分解，取最小奇异值对应的右奇异向量
    cv::Mat w, u, vt;
    cv::SVD::compute(M, w, u, vt);

    // vt 最后一行即解向量
    cv::Mat theta = vt.row(vt.rows - 1);
    double A = theta.at<double>(0);
    double B = theta.at<double>(1);
    double C = theta.at<double>(2);
    double D = theta.at<double>(3);
    double E = theta.at<double>(4);
    double F = theta.at<double>(5);

    params = { A, B, C, D, E, F };

    // 判断是否为椭圆（鉴别式 < 0）
    return (B * B - 4.0 * A * C) < 0.0;
}

// ============================================================
//  主程序
// ============================================================
int main()
{
    // ===== 显示窗口尺寸（按需修改）=====
    const int DISP_W = 1280;
    const int DISP_H = 720;

    // ===================== 1. 读取与显示原图 =====================
    std::string image_path = "DSC_5101.JPG";
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "[错误] 无法读取图像: " << image_path << std::endl;
        std::cerr << "       请确保图片与 task2.exe 在同一目录，或修改 image_path 变量。" << std::endl;
        return -1;
    }
    std::cout << "[信息] 图像尺寸: " << img.cols << " x " << img.rows << std::endl;

    cv::namedWindow("1. Original Image", cv::WINDOW_NORMAL);
    cv::resizeWindow("1. Original Image", DISP_W, DISP_H);
    cv::imshow("1. Original Image", img);

    // ===================== 2. 图像预处理 =====================
    // 转灰度
    cv::Mat gray_img;
    if (img.channels() == 3)
        cv::cvtColor(img, gray_img, cv::COLOR_BGR2GRAY);
    else
        gray_img = img.clone();

    // 去背景（减去固定偏置 10）
    cv::Mat bg_removed;
    cv::subtract(gray_img, cv::Scalar(10), bg_removed);

    // 高斯滤波 5×5, sigma=2
    cv::Mat gaussian_img;
    cv::GaussianBlur(bg_removed, gaussian_img, cv::Size(5, 5), 2.0, 2.0);

    // Otsu 二值化
    cv::Mat bw_img;
    cv::threshold(gaussian_img, bw_img, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    cv::namedWindow("2. Binary Image", cv::WINDOW_NORMAL);
    cv::resizeWindow("2. Binary Image", DISP_W, DISP_H);
    cv::imshow("2. Binary Image", bw_img);

    // ===================== 3. 连通区域分析与规则筛选 =====================
    cv::Mat labels, statsCC, centroidsCC;
    int num_comp = cv::connectedComponentsWithStats(
        bw_img, labels, statsCC, centroidsCC, 8, CV_32S);

    // 存储每个候选区域的信息
    struct RegionInfo {
        cv::Point2d centroid;
        int         area;
        double      mean_intensity;
        double      aspect_ratio;
        std::string status;
        std::vector<double> ellipse_params; // [A B C D E F]
        cv::Point2i local_offset;           // (x1, y1) 局部->全局偏移
        cv::Point2d phys_coord;             // 物理坐标 (mm)
    };

    std::vector<RegionInfo> results(num_comp);
    std::vector<int> candidates_step1;
    std::vector<int> discarded_step1;

    std::cout << "\n[Step 1] 开始规则筛选，连通域总数（含背景）: " << num_comp << std::endl;

    for (int i = 1; i < num_comp; i++) // i=0 是背景，跳过
    {
        int x  = statsCC.at<int>(i, cv::CC_STAT_LEFT);
        int y  = statsCC.at<int>(i, cv::CC_STAT_TOP);
        int w  = statsCC.at<int>(i, cv::CC_STAT_WIDTH);
        int h  = statsCC.at<int>(i, cv::CC_STAT_HEIGHT);
        int ar = statsCC.at<int>(i, cv::CC_STAT_AREA);

        // 加权重心
        double cx = centroidsCC.at<double>(i, 0);
        double cy = centroidsCC.at<double>(i, 1);

        // 计算区域内 gaussian_img 的均值（仅属于该标签的像素）
        cv::Mat mask = (labels == i);
        cv::Scalar mean_val = cv::mean(gaussian_img, mask);
        double mi = mean_val[0];

        // 长宽比
        double aspect = (double)std::max(w, h) / (double)std::max(std::min(w, h), 1);

        results[i].centroid       = cv::Point2d(cx, cy);
        results[i].area           = ar;
        results[i].mean_intensity = mi;
        results[i].aspect_ratio   = aspect;

        // 规则筛选（与 MATLAB 保持一致）
        if (mi < 20.0 || ar > 1000 || ar < 9 || aspect > 5.0) {
            results[i].status = "Discarded_Rule";
            discarded_step1.push_back(i);
        } else {
            results[i].status = "Candidate";
            candidates_step1.push_back(i);
        }
    }

    std::cout << "    候选保留: " << candidates_step1.size()
              << "  规则剔除: " << discarded_step1.size() << std::endl;

    // 可视化第一轮：绿点=保留，红×=剔除
    {
        cv::Mat vis1 = img.clone();
        for (int idx : candidates_step1)
            cv::circle(vis1, results[idx].centroid, 5, cv::Scalar(0, 255, 0), -1);
        for (int idx : discarded_step1)
            cv::drawMarker(vis1, results[idx].centroid,
                           cv::Scalar(0, 0, 255), cv::MARKER_CROSS, 10, 1);
        cv::namedWindow("3. Rule Filter (Green=Keep  Red=Discard)", cv::WINDOW_NORMAL);
        cv::resizeWindow("3. Rule Filter (Green=Keep  Red=Discard)", DISP_W, DISP_H);
        cv::imshow("3. Rule Filter (Green=Keep  Red=Discard)", vis1);
    }

    // ===================== 4. 形状筛选（局部椭圆拟合）=====================
    std::cout << "\n[Step 2] 开始椭圆拟合筛选..." << std::endl;

    std::vector<int> final_valid;
    std::vector<int> discarded_step2;

    const double threshold_dev = 2.0; // 最大几何偏差阈值（像素）
    const int    img_h = bw_img.rows;
    const int    img_w = bw_img.cols;

    for (int i : candidates_step1)
    {
        // 取局部 patch（BoundingBox）
        int x1 = std::max(statsCC.at<int>(i, cv::CC_STAT_LEFT), 0);
        int y1 = std::max(statsCC.at<int>(i, cv::CC_STAT_TOP),  0);
        int w  = statsCC.at<int>(i, cv::CC_STAT_WIDTH);
        int h  = statsCC.at<int>(i, cv::CC_STAT_HEIGHT);
        int x2 = std::min(x1 + w, img_w);
        int y2 = std::min(y1 + h, img_h);

        cv::Rect roi(x1, y1, x2 - x1, y2 - y1);
        cv::Mat local_patch = bw_img(roi);

        // Canny 边缘提取
        cv::Mat edges_local;
        cv::Canny(local_patch, edges_local, 50, 150);

        // 收集边缘点（局部坐标）
        std::vector<cv::Point> edge_pts;
        cv::findNonZero(edges_local, edge_pts);

        if ((int)edge_pts.size() < 6) {
            results[i].status = "Discarded_PointsNotEnough";
            discarded_step2.push_back(i);
            continue;
        }

        // 最小二乘椭圆拟合
        std::vector<double> ep;
        bool valid = fitEllipseLS(edge_pts, ep);

        if (!valid) {
            results[i].status = "Discarded_FitFailed";
            discarded_step2.push_back(i);
            continue;
        }

        results[i].ellipse_params = ep;
        results[i].local_offset   = cv::Point2i(x1, y1);

        // 计算最大几何偏差
        double A = ep[0], B = ep[1], C = ep[2];
        double D = ep[3], E = ep[4], F = ep[5];

        double max_dev = 0.0;
        for (const auto& pt : edge_pts) {
            double x = pt.x, y = pt.y;
            double alg  = A*x*x + B*x*y + C*y*y + D*x + E*y + F;
            double gx   = 2*A*x + B*y + D;
            double gy   = B*x + 2*C*y + E;
            double gnorm = std::sqrt(gx*gx + gy*gy);
            if (gnorm < 1e-10) gnorm = 1e-10;
            double geo = std::abs(alg) / gnorm;
            if (geo > max_dev) max_dev = geo;
        }

        if (max_dev > threshold_dev) {
            results[i].status = "Discarded_Shape";
            discarded_step2.push_back(i);
        } else {
            results[i].status = "Valid";
            final_valid.push_back(i);
        }
    }

    std::cout << "    最终有效目标: " << final_valid.size()
              << "  形状剔除: "     << discarded_step2.size() << std::endl;

    // ===================== 可视化最终结果 =====================
    {
        cv::Mat vis2 = img.clone();

        // 规则剔除 → 红色小叉
        for (int idx : discarded_step1)
            cv::drawMarker(vis2, results[idx].centroid,
                           cv::Scalar(0, 0, 255), cv::MARKER_CROSS, 8, 1);

        // 形状剔除 → 黄色叉
        for (int idx : discarded_step2)
            cv::drawMarker(vis2, results[idx].centroid,
                           cv::Scalar(0, 255, 255), cv::MARKER_CROSS, 10, 2);

        // 最终有效 → 绿色圆圈
        for (int idx : final_valid) {
            cv::circle(vis2, results[idx].centroid, 9,
                       cv::Scalar(0, 255, 0), 2);

            // 在 BoundingBox 内绘制拟合椭圆（近似：用 fitEllipse 做显示用途）
            // ——将局部边缘点转为全局坐标，再调用 OpenCV 自带椭圆绘制
            int bx1 = std::max(statsCC.at<int>(idx, cv::CC_STAT_LEFT), 0);
            int by1 = std::max(statsCC.at<int>(idx, cv::CC_STAT_TOP),  0);
            int bw  = statsCC.at<int>(idx, cv::CC_STAT_WIDTH);
            int bh  = statsCC.at<int>(idx, cv::CC_STAT_HEIGHT);
            int bx2 = std::min(bx1 + bw, img_w);
            int by2 = std::min(by1 + bh, img_h);

            cv::Rect roi2(bx1, by1, bx2 - bx1, by2 - by1);
            cv::Mat patch2 = bw_img(roi2);
            cv::Mat edges2;
            cv::Canny(patch2, edges2, 50, 150);
            std::vector<cv::Point> epts2;
            cv::findNonZero(edges2, epts2);

            // 转全局坐标
            std::vector<cv::Point2f> epts_global;
            for (auto& p : epts2)
                epts_global.push_back(cv::Point2f((float)(p.x + bx1),
                                                   (float)(p.y + by1)));

            if (epts_global.size() >= 5) {
                cv::RotatedRect rr = cv::fitEllipse(epts_global);
                cv::ellipse(vis2, rr, cv::Scalar(0, 255, 0), 1);
            }
        }

        cv::namedWindow("4. Final Result (Green=Valid  Yellow=Shape  Red=Rule)", cv::WINDOW_NORMAL);
        cv::resizeWindow("4. Final Result (Green=Valid  Yellow=Shape  Red=Rule)", DISP_W, DISP_H);
        cv::imshow("4. Final Result (Green=Valid  Yellow=Shape  Red=Rule)", vis2);
    }

    // ===================== 5. 坐标系转换 =====================
    const double pixel_size_x = 0.004878; // mm/pixel
    const double pixel_size_y = 0.004878;
    const double u0 = gray_img.cols / 2.0;
    const double v0 = gray_img.rows / 2.0;

    std::cout << "\n=== 最终识别目标点信息 (前10个) ===" << std::endl;
    std::cout << std::left
              << std::setw(6)  << "ID"
              << std::setw(18) << "Pixel(u, v)"
              << std::setw(16) << "Phys_X(mm)"
              << std::setw(16) << "Phys_Y(mm)" << std::endl;
    std::cout << std::string(56, '-') << std::endl;

    int count = 0;
    for (int k : final_valid) {
        double u    = results[k].centroid.x;
        double v    = results[k].centroid.y;
        double x_mm = (u - u0) * pixel_size_x;
        double y_mm = (v0 - v) * pixel_size_y;
        results[k].phys_coord = cv::Point2d(x_mm, y_mm);

        if (++count <= 10) {
            std::cout << std::left  << std::setw(6) << k
                      << std::fixed << std::setprecision(1)
                      << "[" << std::setw(7) << u << ", " << std::setw(7) << v << "]  "
                      << std::setw(14) << std::setprecision(4) << x_mm
                      << std::setw(14) << y_mm << std::endl;
        }
    }
    std::cout << "\n共识别到有效目标点: " << final_valid.size() << " 个" << std::endl;

    // ===================== 6. 物理坐标系绘图 =====================
    // 用 OpenCV 绘制散点图（以图像方式展示物理坐标分布）
    if (!final_valid.empty()) {
        // 收集物理坐标范围
        double xmin = 1e9, xmax = -1e9, ymin = 1e9, ymax = -1e9;
        for (int k : final_valid) {
            xmin = std::min(xmin, results[k].phys_coord.x);
            xmax = std::max(xmax, results[k].phys_coord.x);
            ymin = std::min(ymin, results[k].phys_coord.y);
            ymax = std::max(ymax, results[k].phys_coord.y);
        }
        // 留边距
        double margin = std::max((xmax - xmin), (ymax - ymin)) * 0.1 + 0.5;
        xmin -= margin; xmax += margin;
        ymin -= margin; ymax += margin;

        int canvas_w = 800, canvas_h = 600;
        cv::Mat canvas(canvas_h, canvas_w, CV_8UC3, cv::Scalar(255, 255, 255));

        // 坐标轴（原点）
        auto toCanvas = [&](double x, double y) -> cv::Point {
            int px = (int)((x - xmin) / (xmax - xmin) * (canvas_w - 80)) + 40;
            int py = canvas_h - 40 - (int)((y - ymin) / (ymax - ymin) * (canvas_h - 80));
            return cv::Point(px, py);
        };

        // 网格线 & 轴
        cv::line(canvas, toCanvas(xmin, 0), toCanvas(xmax, 0),
                 cv::Scalar(180, 180, 180), 1);
        cv::line(canvas, toCanvas(0, ymin), toCanvas(0, ymax),
                 cv::Scalar(180, 180, 180), 1);

        // 原点
        cv::drawMarker(canvas, toCanvas(0, 0),
                       cv::Scalar(0, 0, 0), cv::MARKER_CROSS, 20, 2);

        // 散点
        for (int k : final_valid) {
            cv::circle(canvas, toCanvas(results[k].phys_coord.x,
                                        results[k].phys_coord.y),
                       4, cv::Scalar(255, 80, 0), -1);
        }

        // 标注
        cv::putText(canvas, "Physical Coordinate Distribution (mm)",
                    cv::Point(20, 25), cv::FONT_HERSHEY_SIMPLEX,
                    0.6, cv::Scalar(50, 50, 50), 1);
        cv::putText(canvas, "X (mm)", cv::Point(canvas_w - 70, canvas_h / 2 + 15),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(100, 100, 100), 1);
        cv::putText(canvas, "Y (mm)", cv::Point(5, 45),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(100, 100, 100), 1);

        cv::namedWindow("5. Physical Coordinate (mm)", cv::WINDOW_NORMAL);
        cv::resizeWindow("5. Physical Coordinate (mm)", DISP_W, DISP_H);
        cv::imshow("5. Physical Coordinate (mm)", canvas);
    }

    std::cout << "\n[Done] Press ESC to exit, or close all windows manually." << std::endl;

    // 循环等待，直到用户按 ESC 键才退出，不会因为鼠标点击而意外关闭
    while (true) {
        int key = cv::waitKey(100); // 每 100ms 检测一次
        if (key == 27) break;       // ESC 键退出

        // 检测是否所有窗口都已被手动关闭
        // getWindowProperty 返回 -1 表示窗口已关闭
        bool any_open = false;
        std::vector<std::string> win_names = {
            "1. Original Image",
            "2. Binary Image",
            "3. Rule Filter (Green=Keep  Red=Discard)",
            "4. Final Result (Green=Valid  Yellow=Shape  Red=Rule)",
            "5. Physical Coordinate (mm)"
        };
        for (auto& name : win_names) {
            if (cv::getWindowProperty(name, cv::WND_PROP_VISIBLE) >= 1) {
                any_open = true;
                break;
            }
        }
        if (!any_open) break; // 所有窗口都关了就退出
    }

    cv::destroyAllWindows();
    return 0;
}