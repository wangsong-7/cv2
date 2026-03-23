
#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <climits>

// ─────────────────────────────────────────────────────────────
//  工具函数：弹出命名窗口并等待任意键
//  注意：点击图像窗口后按键盘任意键继续，不要点 × 关闭！
// ─────────────────────────────────────────────────────────────
static void showAndWait(const std::string& winName, const cv::Mat& img)
{
    cv::namedWindow(winName, cv::WINDOW_NORMAL);
    cv::imshow(winName, img);
    std::cout << "[点击图像窗口后按任意键继续 → " << winName << "]\n";
    cv::waitKey(0);
    cv::destroyWindow(winName);
}

// ─────────────────────────────────────────────────────────────
//  存储每个目标点的信息
// ─────────────────────────────────────────────────────────────
struct TargetPoint {
    cv::Point2f topLeft;
    cv::Point2f bottomRight;
    int         area;
    cv::Point2f centroid;
};

int main()
{
    // 设置控制台编码为 UTF-8，解决中文乱码
    system("chcp 65001 > nul");

    std::cout << "程序启动...\n";

    // ══════════════════════════════════════════════════════════
    //  Step 1. 读取原始图像
    //  !! 把下面路径改成你图片的实际路径，路径中不要有中文 !!
    // ══════════════════════════════════════════════════════════
    const std::string imagePath = "D:\\cv2\\Measurement Task\\DSC_5101.JPG";

    std::cout << "正在读取图片：" << imagePath << "\n";
    cv::Mat img = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "错误：找不到图片！请检查路径：" << imagePath << "\n";
        system("pause");
        return -1;
    }
    std::cout << "图像尺寸：" << img.cols << " x " << img.rows
              << "，通道数：" << img.channels() << "\n";

    showAndWait("1. Yuan Shi Tu Xiang", img);


    // ══════════════════════════════════════════════════════════
    //  Step 2. 转换为灰度图
    // ══════════════════════════════════════════════════════════
    std::cout << "Step 2: 转换灰度图...\n";
    cv::Mat grayImg;
    if (img.channels() == 3)
        cv::cvtColor(img, grayImg, cv::COLOR_BGR2GRAY);
    else
        grayImg = img.clone();

    showAndWait("2. Hui Du Tu", grayImg);


    // ══════════════════════════════════════════════════════════
    //  Step 3. 去除背景（每个像素 -10，饱和减法）
    // ══════════════════════════════════════════════════════════
    std::cout << "Step 3: 去除背景...\n";
    cv::Mat bgRemoved;
    cv::subtract(grayImg, cv::Scalar(10), bgRemoved);

    showAndWait("3. Qu Chu Bei Jing", bgRemoved);


    // ══════════════════════════════════════════════════════════
    //  Step 4. 高斯滤波（5×5，sigma=2）
    // ══════════════════════════════════════════════════════════
    std::cout << "Step 4: 高斯滤波...\n";
    cv::Mat gaussianImg;
    cv::GaussianBlur(bgRemoved, gaussianImg, cv::Size(5, 5), 2.0, 2.0,
                     cv::BORDER_REPLICATE);

    showAndWait("4. Gao Si Lv Bo", gaussianImg);


    // ══════════════════════════════════════════════════════════
    //  Step 5. 二值化（阈值 = 0.1 × 255 ≈ 25）
    // ══════════════════════════════════════════════════════════
    std::cout << "Step 5: 二值化...\n";
    cv::Mat bwImg;
    cv::threshold(gaussianImg, bwImg, 0.1 * 255.0, 255, cv::THRESH_BINARY);

    showAndWait("5. Er Zhi Hua", bwImg);


    // ══════════════════════════════════════════════════════════
    //  Step 6. 连通区域分析并用伪彩色显示
    // ══════════════════════════════════════════════════════════
    std::cout << "Step 6: 连通区域分析...\n";
    cv::Mat labelMap;
    int numLabels = cv::connectedComponents(bwImg, labelMap, 8, CV_32S);

    // 生成伪彩色图
    cv::Mat colorLabel(labelMap.size(), CV_8UC3, cv::Vec3b(0, 0, 0));
    std::vector<cv::Vec3b> colors(numLabels);
    colors[0] = cv::Vec3b(0, 0, 0);
    cv::RNG rng(42);
    for (int k = 1; k < numLabels; ++k)
        colors[k] = cv::Vec3b(rng.uniform(50, 255),
                              rng.uniform(50, 255),
                              rng.uniform(50, 255));
    for (int r = 0; r < labelMap.rows; ++r)
        for (int c = 0; c < labelMap.cols; ++c)
            colorLabel.at<cv::Vec3b>(r, c) = colors[labelMap.at<int>(r, c)];

    std::cout << "共检测到连通区域（不含背景）：" << numLabels - 1 << " 个\n";
    showAndWait("6. Lian Tong Qu Yu", colorLabel);


    // ══════════════════════════════════════════════════════════
    //  Step 7. 灰度重心法（优化版：单次遍历全图，适合大图）
    //
    //  公式：
    //    x = Σ(xi * f(xi,yi)) / Σf(xi,yi)
    //    y = Σ(yi * f(xi,yi)) / Σf(xi,yi)
    // ══════════════════════════════════════════════════════════
    std::cout << "Step 7: 计算灰度重心（大图可能需要几秒）...\n";

    cv::Mat resultImg;
    cv::cvtColor(gaussianImg, resultImg, cv::COLOR_GRAY2BGR);

    // 一次性收集所有连通域数据
    struct LabelData {
        double sumF  = 0;
        double sumXF = 0;
        double sumYF = 0;
        int minX = INT_MAX, minY = INT_MAX;
        int maxX = 0,       maxY = 0;
        int count = 0;
    };
    std::vector<LabelData> labelData(numLabels);

    // 只遍历一次全图
    for (int r = 0; r < labelMap.rows; ++r) {
        for (int c = 0; c < labelMap.cols; ++c) {
            int lbl = labelMap.at<int>(r, c);
            if (lbl == 0) continue;
            double f = static_cast<double>(gaussianImg.at<uchar>(r, c));
            auto& ld = labelData[lbl];
            ld.sumF  += f;
            ld.sumXF += c * f;
            ld.sumYF += r * f;
            ld.minX = std::min(ld.minX, c);
            ld.minY = std::min(ld.minY, r);
            ld.maxX = std::max(ld.maxX, c);
            ld.maxY = std::max(ld.maxY, r);
            ld.count++;
        }
    }

    std::vector<TargetPoint> targetPoints;

    for (int lbl = 1; lbl < numLabels; ++lbl) {
        auto& ld = labelData[lbl];
        if (ld.count == 0) continue;

        // 计算灰度重心
        cv::Point2f centroid;
        if (ld.sumF > 0) {
            centroid.x = static_cast<float>(ld.sumXF / ld.sumF);
            centroid.y = static_cast<float>(ld.sumYF / ld.sumF);
        } else {
            centroid.x = (ld.minX + ld.maxX) / 2.0f;
            centroid.y = (ld.minY + ld.maxY) / 2.0f;
        }

        cv::Rect bb(ld.minX, ld.minY,
                    ld.maxX - ld.minX,
                    ld.maxY - ld.minY);

        // 绘制红色包围盒
        cv::rectangle(resultImg, bb, cv::Scalar(0, 0, 255), 2);

        // 绘制绿色十字重心
        const int armLen = 8;
        cv::line(resultImg,
                 cv::Point(cvRound(centroid.x) - armLen, cvRound(centroid.y)),
                 cv::Point(cvRound(centroid.x) + armLen, cvRound(centroid.y)),
                 cv::Scalar(0, 255, 0), 2);
        cv::line(resultImg,
                 cv::Point(cvRound(centroid.x), cvRound(centroid.y) - armLen),
                 cv::Point(cvRound(centroid.x), cvRound(centroid.y) + armLen),
                 cv::Scalar(0, 255, 0), 2);

        // 存储结果
        TargetPoint tp;
        tp.topLeft     = cv::Point2f(static_cast<float>(ld.minX),
                                     static_cast<float>(ld.minY));
        tp.bottomRight = cv::Point2f(static_cast<float>(ld.maxX),
                                     static_cast<float>(ld.maxY));
        tp.area        = ld.count;
        tp.centroid    = centroid;
        targetPoints.push_back(tp);
    }

    showAndWait("7. Mu Biao Dian (Red:BBox  Green:Centroid)", resultImg);


    // ══════════════════════════════════════════════════════════
    //  Step 8. 控制台输出结果
    // ══════════════════════════════════════════════════════════
    std::cout << "\n检测到的目标点信息（基于灰度重心法）：\n";
    if (targetPoints.empty()) {
        std::cout << "  未检测到任何目标点。\n";
    } else {
        std::cout << std::fixed << std::setprecision(2);
        std::cout << std::left
                  << std::setw(6)  << "编号"
                  << std::setw(18) << "TopLeft(x,y)"
                  << std::setw(20) << "BottomRight(x,y)"
                  << std::setw(10) << "Area"
                  << std::setw(20) << "Centroid(x,y)"
                  << "\n";
        std::cout << std::string(74, '-') << "\n";
        for (size_t i = 0; i < targetPoints.size(); ++i) {
            const auto& tp = targetPoints[i];
            std::cout
                << std::setw(6)  << i + 1
                << std::setw(18) << ("(" + std::to_string((int)tp.topLeft.x)
                                         + "," + std::to_string((int)tp.topLeft.y) + ")")
                << std::setw(20) << ("(" + std::to_string((int)tp.bottomRight.x)
                                         + "," + std::to_string((int)tp.bottomRight.y) + ")")
                << std::setw(10) << tp.area
                << std::setw(20) << ("(" + std::to_string(tp.centroid.x)
                                         + "," + std::to_string(tp.centroid.y) + ")")
                << "\n";
        }
    }

    std::cout << "\n全部完成！\n";
    cv::destroyAllWindows();
    system("pause");
    return 0;
}