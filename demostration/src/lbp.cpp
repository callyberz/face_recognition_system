#include "lbp.h"
#include <iostream>

using namespace cv;
using namespace std;

void olbp(Mat _src, Mat _dst) {

    // calculate patterns
    for(int i=1;i<_src.rows-1;i++) {
        for(int j=1;j<_src.cols-1;j++) {
            int center = _src.at<uchar>(i,j);
            int code = 0;
            if (_src.at<uchar>(i-1,j-1) >= center) {
            	code = code + 128;
            }
            if (_src.at<uchar>(i-1,j) >= center) {
            	code = code + 64;
            }
            if (_src.at<uchar>(i-1,j+1) >= center) {
            	code = code + 32;
            }
            if (_src.at<uchar>(i,j+1) >= center) {
            	code = code + 16;
            }
            if (_src.at<uchar>(i+1,j+1) >= center) {
            	code = code + 8;
            }
            if (_src.at<uchar>(i+1,j) >= center) {
            	code = code + 4;
            }
            if (_src.at<uchar>(i+1,j-1) >= center) {
            	code = code + 2;
            }
            if (_src.at<uchar>(i,j-1) >= center) {
            	code = code + 1;
            }
            _dst.at<uchar>(i-1,j-1) = code;
        }
    }
}