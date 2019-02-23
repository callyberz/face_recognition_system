#pragma once
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <vector>

cv::Mat getEigenFace(cv::Mat eigenVec, cv::Mat A, unsigned int numComp);

void bruteForceEigen(cv::Mat covMat);

cv::Mat set2matrix(std::vector<cv::Mat> &image_set);

double classify(cv::Mat trainProjection, cv::Mat testProjection, std::vector<int> true_index, int setID, std::vector<int> &TP_index);

/* Read image set from CSV file*/
void read_csv(const std::string& filename, std::vector<cv::Mat>& images, std::vector<int>& labels);

double classifylabel(cv::Mat trainProjection, std::vector<int> trainlabel, cv::Mat testProjection, std::vector<int> testlabel);