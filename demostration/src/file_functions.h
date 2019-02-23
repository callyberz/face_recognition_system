#pragma once
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp" 
#include <iostream>
#include <string.h>
#include <sstream>

#ifdef __linux
template <typename T>
std::string to_string(T value){
	//create an output string stream
	std::ostringstream os ;
	//throw the value into the string stream
	os << value ;
	//convert the string stream into a string and return
	return os.str() ;
}
#endif

/* writeFile - output file and save it */
void writeFile(cv::Mat data, std::string dataName);

// Turns a given matrix into its grayscale representation. ***for image representations
cv::Mat toGrayscale(const cv::Mat& src, int dtype = CV_8UC1);
void read_csv(const std::string& filename, std::vector<cv::Mat>& images, std::vector<int>& labels);
void read_csv_lfw(const std::string& filename, std::vector<cv::Mat>& images, std::vector<std::string>& labels);