#include <iostream>
#include <opencv2\opencv.hpp>
using namespace std;
using namespace cv;
 
 
int main()
{
    string path = "C:\\Users\\86159\\Desktop\\6521526b673737b515f0313cf50d8b7f.jpg";
    cv::Mat img = imread(path);
    imshow("img",img);
    waitKey(0);
   cv::waitKey(0);  // 按任意键关闭
return 0;
}

