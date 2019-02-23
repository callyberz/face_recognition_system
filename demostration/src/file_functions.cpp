#include "file_functions.h"
#include "set_definitions.h"
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

/* writeFile - output file and save it */
void writeFile(Mat data, string dataName){
	FileStorage f("output/" + dataName + ".yml", FileStorage::WRITE);
	f << dataName << data; f.release();
}

Mat toGrayscale(const cv::Mat& src, int dtype) {
    // only allow one channel
    if(src.channels() != 1) {
        string error_message = format("Only Matrices with one channel are supported. Expected 1, but was %d.", src.channels());
        CV_Error(CV_StsBadArg, error_message);
    }
    // create and return normalized image
    Mat dst;
    cv::normalize(src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
    return dst;
}

void read_csv(const std::string& filename, std::vector<cv::Mat>& images, std::vector<int>& labels) {
	ifstream file(filename.c_str(), ifstream::in);
	if(!file)
		throw exception();
	string line, path, classlabel;
	// for each line
	while (getline(file, line)) {
		// get current line
		stringstream liness(line);
		// split line
		getline(liness, path, ';');
		getline(liness, classlabel);
		// push pack the data
		images.push_back(imread(path,0));
		labels.push_back(atoi(classlabel.c_str()));
	}
}


void read_csv_lfw(const std::string& filename, std::vector<cv::Mat>& images, std::vector<string>& labels) {
	ifstream file(filename.c_str(), ifstream::in);
	string suffux = "/Users/calvinlee/Desktop/FYP_ALL/Programming/data/lfw/faces/";
	if(!file)
		throw exception();
	string line, path, classlabel;
	// for each line
	while (getline(file, line)) {
		line.pop_back();
		// get current line
		stringstream liness(line);
		//concatenate a path to the img
		path = suffux + line + ".pgm";
		// split line
		getline(liness, classlabel, '0');
		//remove the '_' in the last index
		classlabel.pop_back();
		// push pack the data
		images.push_back(imread(path,0));
		labels.push_back(classlabel);
	}
}

