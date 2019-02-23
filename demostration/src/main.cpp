#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "file_functions.h"
#include "eigenfaces.h"
#include "set_definitions.h"
#include "lbp.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <string.h>

#define CVUI_IMPLEMENTATION
#include "cvui.h"

using namespace std;
using namespace cv;
using namespace cvui;

#define WINDOW_NAME "Final Year Project Demostration"

void loadORLimages();
void loadLFWimages();
void lbp();
void eigenfacesPCA();
void fisherfacesLDA();
string fn_csv; string fn_csv_test;
vector<Mat> trainimages; vector<int> trainlabels;
vector<Mat> testimages; vector<int> testlabels;

int main(int argc, const char *argv[])
{
	Mat frame = cv::Mat(350, 700, CV_8UC3);
	Mat img1 = cv::imread("/Users/calvinlee/Desktop/orl_faces/at/s1/1.pgm", cv::IMREAD_COLOR);
	Mat img2 = cv::imread("/Users/calvinlee/eigenfaces/data/lfw/faces/Aaron_Eckhart_0001.pgm", cv::IMREAD_COLOR);
	int count = 5;

	// Init cvui and tell it to create a OpenCV window, i.e. cv::namedWindow(WINDOW_NAME).
	cvui::init(WINDOW_NAME);

	while (true) {
		// Fill the frame with a nice color
		frame = cv::Scalar(49, 52, 49);

		cvui::text(frame, 10, 10, "This is a simple face recognition system");
		cvui::counter(frame, 100, 100, &count);

		cvui::beginRow(frame, 20, 30, 200, 50);
			if (cvui::button(120, 30, "Train ORL images")) {
				cout<<"Number of training images is "<<count<<endl;
				srand(time(NULL));
				int randnum_int = (rand()%5)+1;
				string randnum1 = to_string(randnum_int);
				srand (4); randnum_int = (rand()%5)+1;
				// string randnum2 = to_string(randnum_int);
				cout<<"Using training image set "<<randnum1<<endl;
				// cout<<randnum2<<endl;
				fn_csv = "/Users/calvinlee/Desktop/FYP_ALL/Programming/data/at_alltxt/at_"+to_string(count)+"trainingimg"+randnum1+".txt";
				// fn_csv_test = "/Users/calvinlee/eigenfaces/data/at_test.txt";
				fn_csv_test = "/Users/calvinlee/Desktop/FYP_ALL/Programming/data/at_alltxt/at_test.txt";
				loadORLimages();
			};
			if (cvui::button(120, 30, "Train LFW images")) {
				loadLFWimages();
			};
			cvui::image(img1);
			cvui::text("ORL database     ");
			cvui::image(img2);
			cvui::text("LFW database");
		cvui::endRow();

		cvui::beginRow(frame, 20, 150, 200, 50);
		cvui::text("Choose any one algorithm to classify");
		cvui::endRow();

		cvui::beginRow(frame, 20, 170, 200, 50);
			if (cvui::button(100, 30, "Eigenfaces")) {
				eigenfacesPCA();
			};
			if (cvui::button(100, 30, "Fisherfaces")) {
				fisherfacesLDA();
			};
			if (cvui::button(100, 30, "LBP")) {
				lbp();
			};
			if (cvui::button(100, 30, "&Quit")) {
				cout<<"Quit now!"<<endl;
				break;
			}
		cvui::endRow();

		// This function must be called *AFTER* all UI components. It does
		// all the behind the scenes magic to handle mouse clicks, etc.
		cvui::update();

		// Show everything on the screen
		cvui::imshow(WINDOW_NAME, frame);

		// Check if ESC key was pressed
		if (cv::waitKey(20) == 27) {
			break;
		}
	}

	return 0;
}


void loadORLimages() {
	cout<<"Loading LFW database"<<endl;
	cout<<"Training the images"<<endl;

    try {
        read_csv(fn_csv, trainimages, trainlabels);
        read_csv(fn_csv_test, testimages, testlabels);
    } catch(exception& e) {
        cerr << "Error opening file \"" << fn_csv << "\"." << endl;
        exit(1);
    }
    // Show the 1st image in ORL database
    // Mat col1 = trainimages[0].reshape(0, trainimages[0].rows);
    // cv::imshow("ORL image", col1);
}

void loadLFWimages() {
	cout<<"Loading LFW database"<<endl;
	cout<<"Training the images"<<endl;
    vector<string> trainlabels_lfw;
    // // // ------ Get images from LFW ------ // // //
	string fn_csv_lfw = "/Users/calvinlee/Desktop/FYP_ALL/Programming/data/lfw/lists/01_train_same_copy.txt";
	try {
	read_csv_lfw(fn_csv_lfw, trainimages, trainlabels_lfw);
} catch(exception& e){
	cerr << "Error opening file \"" << fn_csv_lfw << "\"." << endl;
        exit(1);
    }
    trainimages.pop_back();
    trainlabels_lfw.pop_back();

    int trainingindex = 0;
    for (int i=0; i<trainlabels_lfw.size(); i++) {
    	string currentlabel = trainlabels_lfw[i];

    	if (currentlabel == trainlabels_lfw[i+1]) {
    		trainlabels.push_back(trainingindex);
    	}
    	else {
    		trainingindex++;
    		trainlabels.push_back(trainingindex);
    	}
    }

    vector<string> testlabels_lfw;
    // // // ------ Get images from LFW ------ // // //
	string fn_csv_lfw2 = "/Users/calvinlee/Desktop/FYP_ALL/Programming/data/lfw/lists/02_test_same_copy.txt";
	try {
	read_csv_lfw(fn_csv_lfw2, testimages, testlabels_lfw);
} catch(exception& e){
	cerr << "Error opening file \"" << fn_csv_lfw << "\"." << endl;
        exit(1);
    }
    testimages.pop_back();
    testlabels_lfw.pop_back();

    int testingindex = 0;
    for (int i=0; i<testlabels_lfw.size(); i++) {
    	string currentlabel = testlabels_lfw[i];

    	if (currentlabel == testlabels_lfw[i+1]) {
    		testlabels.push_back(testingindex);
    	}
    	else {
    		testingindex++;
    		testlabels.push_back(testingindex);
    	}
    }
}

void eigenfacesPCA() {
	if (!trainimages.empty()) {
	cout<<"Performing eigenfaces (PCA)"<<endl;
    Mat trainMatrix = set2matrix(trainimages);
    Mat testMatrix = set2matrix(testimages);
    Mat meanVector(trainMatrix.rows, 1, CV_32FC1);
	reduce(trainMatrix, meanVector, 1, CV_REDUCE_AVG);
	cv::imshow("Mean Face",toGrayscale(meanVector.reshape(0, trainimages[0].rows)));
	Mat meanMatrix;
	repeat(meanVector, 1, trainMatrix.cols, meanMatrix);

	// // // ---------------------------------------------------------------------------// // //
	// // // ------ Step 3: Substract every images in the dataset from Mean face ------ // // //
	// // // ---------------------------------------------------------------------------// // //
	Mat demeanedFaces(trainMatrix.rows, trainMatrix.cols, CV_32FC1);
	demeanedFaces = trainMatrix - meanMatrix;
	Mat my_train_reconstruct = demeanedFaces + meanMatrix;
	Mat demeanedFacesT;	transpose(demeanedFaces, demeanedFacesT);	// 70 x 2500
	// Mat my_covMat = demeanedFaces * demeanedFacesT;			// [2500 x 70]*[70 x 2500] = [2500 x 2500]
	Mat coMatrixB = demeanedFacesT * demeanedFaces;			// [70 x 2500]*[2500 x 70] = [70 x 70]

	// // // ----------------------------------------------------------------------------// // //
	// // // ------ Step 4: Find eigenvalues & eigenvectors of covariance matrix ------ // // //
	// // // --------------------------------------------------------------------------// // //
	Mat eigenValue, eigenVector;
	eigen(coMatrixB, eigenValue, eigenVector);
	Mat eigenFacesT = getEigenFace(eigenVector, demeanedFaces, 10);	// d x 2500: Perform v_i=A*u_i
		vector<Mat> eigenFacesT_vec;
		vector<Mat> eigenFacesT_vec_normalized;
		Mat all_eigenFacesMat;	// display eigenfaces altogether
		for (int i = 0; i < 10; i++){
			Mat eigenFacesT_normalized;
			eigenFacesT_vec.push_back((eigenFacesT.row(i)).reshape(0, trainimages[0].rows)); //put in the vector first
			eigenFacesT_normalized = toGrayscale(eigenFacesT_vec[i]);
			eigenFacesT_vec_normalized.push_back(eigenFacesT_normalized);
		}

	// // // ------ Show all 10 eigenfaces together ------ // // //
	hconcat(eigenFacesT_vec_normalized, all_eigenFacesMat);
	cv::imshow("All 10 eigenfaces" , all_eigenFacesMat);

	// // // ------ Normalize the eignfaces ------ // // //
	Mat eigenFacesNormT(10, trainMatrix.rows, CV_32FC1);
	for (int i = 0; i < eigenFacesT.cols; i++){	// Normalizar la transformación v_i = A * u_i
		normalize(eigenFacesT.col(i), eigenFacesNormT.col(i));
	}
	Mat eigenFacesNorm(trainMatrix.rows, 10, CV_32FC1);			// 10340 x d
	transpose(eigenFacesNormT, eigenFacesNorm);
	Mat groundProj = eigenFacesNormT * demeanedFaces; 	// base projection

	// // // --------------------------------------// // //
	// // // ------ Step 5: Classify image ------ // // //
	// // // ------------------------------------// // //
	Mat meanMatrix_test;
	repeat(meanVector, 1, testMatrix.cols, meanMatrix_test);
	Mat demeanedFaces_test = testMatrix - meanMatrix_test;
	Mat projection1 = eigenFacesNormT * demeanedFaces_test;

	double errorrate_PCA = classifylabel(groundProj, trainlabels, projection1, testlabels);
	cout<<"PCA error rate = "<<errorrate_PCA<<endl;
	}
	else cout<<"No training images"<<endl;
}

void fisherfacesLDA() {
	if (!trainimages.empty()) {
		cout<<"Performing Fisherfaces"<<endl;
		Ptr<FaceRecognizer> fisher_model = createFisherFaceRecognizer();
		fisher_model->train(trainimages, trainlabels);
		int positive = 0; int negative = 0; double errorrate_fisher;
		for (int i=0; i<testimages.size(); i++) {
			int predictedLabel_fisher = fisher_model->predict(testimages[i]);
			// cout << "actual = " << testlabels[i] << " fisherface predict = "<< predictedLabel_fisher<<endl;
			if (testlabels[i] == predictedLabel_fisher) {
				positive++;
			}
			else {
				negative++;
			}
			errorrate_fisher = (double)negative/testlabels.size();
		}
		cout<< "Fisher error rate = "<<errorrate_fisher<<endl;
	}
	else cout<<"No training images"<<endl;
}

void lbp() {
	if (!trainimages.empty()) {
		cout<<"Performing Local Binary Pattern"<<endl;
		Ptr<FaceRecognizer> face_model = createLBPHFaceRecognizer();
		face_model->train(trainimages, trainlabels);
		int pos = 0; int neg = 0; double errorrate_LBP;
		for (int i=0; i<testimages.size(); i++) {
			int predictedLabel = face_model->predict(testimages[i]);
			// cout << "actual = " << testlabels[i] << " LBP predict = "<< predictedLabel<<endl;
			if (testlabels[i] == predictedLabel) {
				pos++;
			}
			else {
				neg++;
			}
			errorrate_LBP = (double) neg/testlabels.size();
		}
		cout<< "LBP error rate = " << errorrate_LBP <<endl << endl;
	}
	else cout<<"No training images"<<endl;

}


