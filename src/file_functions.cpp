#include "file_functions.h"
#include "set_definitions.h"
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

/* getImage - Carga la imagen {num} de la persona {id} */
string getImage(int id, int num){
	if (id < 10){
		if (num < 10) {
			return "/Users/calvinlee/eigenfaces/data/faces/person0" + to_string(id) + "_0" + to_string(num) + ".png";
		}
		else {
			return "/Users/calvinlee/eigenfaces/data/faces/person0" + to_string(id) + "_" + to_string(num) + ".png";
		}
	}
	else{
		if (num < 10) {
			return "/Users/calvinlee/eigenfaces/data/faces/person" + to_string(id) + "_0" + to_string(num) + ".png";
		}
		else {
			return "/Users/calvinlee/eigenfaces/data/faces/person" + to_string(id) + "_" + to_string(num) + ".png";
		}
	}
}

/* loadImage - Permite cargar imagenes iterativamente */
Mat loadImage(int &id, int &num, bool &init, int i, int setID){
	Mat output;

	int numFloor, numCeil, set_size;

	switch (setID){
	case 1:
            // cout<<"1"<<endl;
		numFloor = 0; numCeil = 7;
		set_size = SET1_SIZE;
		break;
	case 2:
            // cout<<"2"<<endl;
		numFloor = 7; numCeil = 19;
		set_size = SET2_SIZE;
		break;
	case 3:
            // cout<<"3"<<endl;
		numFloor = 19; numCeil = 31;
		set_size = SET3_SIZE;
		break;
	case 4:
            // cout<<"4"<<endl;
		numFloor = 31; numCeil = 45;
		set_size = SET4_SIZE;
		break;
	case 5:
            // cout<<"5"<<endl;
		numFloor = 45; numCeil = 64;
		set_size = SET5_SIZE;
		break;
	default:
		cout << "Debe escoger Sets entre 1 y 5" << endl;
		return output;
	}

	if (!init){ num = numFloor; init = true; }

	if (id < 10){
		if (num < numCeil && num >= numFloor){
			output = imread(getImage(id + 1, num + 1), 0);
			//cout << "Imagen[" + to_string(i) + "] = " + getImage(id + 1, num + 1) << endl;
			num++;
		}
		else{
			num = numFloor;
			id++;

			output = imread(getImage(id + 1, num + 1), 0);
			//cout << "Imagen[" + to_string(i) + "] = " + getImage(id + 1, num + 1) << endl;
			num++;
		}
	}
	else{
		id = 0;

		if (num < numCeil && num >= numFloor){
			output = imread(getImage(id + 1, num + 1), 0);
			//cout << "Imagen[" + to_string(i) + "] = " + getImage(id + 1, num + 1) << endl;
			num++;
		}
		else{
			num = numFloor;
			id++;

			output = imread(getImage(id + 1, num + 1), 0);
			//cout << "Imagen[" + to_string(i) + "] = " + getImage(id + 1, num + 1) << endl;
			num++;
		}
	}
	return output;
}

/* writeFile - Escribe archivos .yml con nombre {dataName}  y matriz {data}*/
void writeFile(Mat data, string dataName){
	FileStorage f("output/" + dataName + ".yml", FileStorage::WRITE);
	f << dataName << data; f.release();
}

/* loadSet - Carga el Set de imagenes {setID} */
vector<Mat> loadSet(int setID){
	vector<Mat> image_set;
	int set_size;
	switch (setID){
	case 1: set_size = SET1_SIZE; break;
	case 2: set_size = SET2_SIZE; break;
	case 3: set_size = SET3_SIZE; break;
	case 4: set_size = SET4_SIZE; break;
	case 5: set_size = SET5_SIZE; break;
	}

	int id = 0; int num = 0; bool init = false;
    for (int i = 0; i < set_size; i++){
		image_set.push_back(loadImage(id, num, init, i, setID));
    }

	return image_set;
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

