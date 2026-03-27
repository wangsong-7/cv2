#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <exception>

using namespace std;
using namespace cv;

// 简单日志输出宏定义
#define LOG_INFO(msg)  cout << "[INFO] " << msg << endl
#define LOG_ERROR(msg) cerr << "[ERROR] " << msg << endl

int main() {
    // 1. 基本参数设置 (王老大，这里请根据您的实际标定板修改)
    // 注意：Size(w, h) 指的是内角点的数量，比如 10x7 的格子，内角点是 9x6
    Size boardSize(11, 7); 
    string inputFolder = "D:/photo test/*.png"; // 存放图片的文件夹路径
    string outputFolder ="D:/photo test/output/";    // 输出结果的路径

    // 2. 批量读取图片 (使用 cv::glob 遍历文件夹)
    vector<String> imageFiles ;
    glob(inputFolder, imageFiles, false);

    if (imageFiles.empty()) {
        LOG_ERROR("未找到任何图片，请检查路径: " + inputFolder);
        return -1;
    }
    LOG_INFO("成功读取图片数量: " + to_string(imageFiles.size()));

    // 3. 遍历处理每一张图片
    for (size_t i = 0; i < imageFiles.size(); i++) {
        try {
            Mat img = imread(imageFiles[i]);
            if (img.empty()) {
                throw runtime_error("无法读取图片: " + imageFiles[i]);
            }

            Mat gray;
            cvtColor(img, gray, COLOR_BGR2GRAY);

            // 存放检测到的角点
            vector<Point2f> corners;

            // 4. 检测棋盘格角点
            bool found = findChessboardCorners(gray, boardSize, corners,
                CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE | CALIB_CB_FAST_CHECK);

            if (found) {
                // 5. 亚像素优化 (极大提升精度)
                // 窗口大小设置，通常为 11x11，停止条件为迭代 30 次或精度达到 0.1
                TermCriteria criteria(TermCriteria::EPS | TermCriteria::MAX_ITER, 30, 0.1);
                cornerSubPix(gray, corners, Size(11, 11), Size(-1, -1), criteria);

                // 6. 可视化：绘制角点
                drawChessboardCorners(img, boardSize, corners, found);

                // 7. 保存结果图
                // 提取文件名以便保存
                size_t lastSlash = imageFiles[i].find_last_of("\\/");
                string fileName = imageFiles[i].substr(lastSlash + 1);
                string savePath = outputFolder + "res_" + fileName;
                
                imwrite(savePath, img);
                LOG_INFO("处理成功并保存: " + savePath);
            } else {
                LOG_ERROR("未检测到完整的棋盘格角点: " + imageFiles[i]);
            }
        } 
        // 8. 异常处理
        catch (const exception& e) {
            LOG_ERROR("处理第 " + to_string(i) + " 张图片时发生异常: " + e.what());
            // 发生异常不退出，继续处理下一张
            continue; 
        }
    }

    LOG_INFO("王老大，本周图像批量处理任务执行完毕！");
    return 0;
}