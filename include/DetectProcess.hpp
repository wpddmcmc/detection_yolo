extern "C"
{
#include "darknet.h"
}
#pragma once
#include "opencv2/opencv.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include "setting.hpp"

using namespace cv;
using namespace std;

class DetectProcess{
  public:
    DetectProcess(Settings *settings)
    {
        _settings = settings;
    }
    void ImageReader();
    void ImageProcesser();
    void Image2Mat(image RefImg, Mat &Img);
    void Mat2Image(Mat RefImg, image *im);
    void Detecter(Mat &src);

  public:
    Settings * _settings;
};