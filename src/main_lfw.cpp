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

using namespace std;
using namespace cv;

int main(int argc, char *argv[]){
	cout<<"The programme is beginning soon"<<endl;
	int d; //default 10 pricinple components
	if(argc > 1){
		d = atoi(argv[1]);
	}
	else{
		d = 10;
	}
	srand(time(NULL));
	int randnum_int = (rand()%10)+1;
	string randnum1 = to_string(randnum_int);
	string randnum2 = to_string((randnum_int%10));
	// cout<<"It's using " + randnum1 + " training set" <<endl;
	// cout<<"It's using " + randnum2 + " testing set" <<endl;
	// // // -----------------------------------------------------// // //
	// // // ------ Step 1: Get the dataset by reading CSV ------ // // //-------------------------// // //
	vector<Mat> trainimages;
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
    vector<int> trainlabels;

    int trainingindex = 0;
    for (int i=0; i<trainlabels_lfw.size(); i++) {
    	string currentlabel = trainlabels_lfw[i];

    	if (currentlabel == trainlabels_lfw[i+1]) {
    		trainlabels.push_back(trainingindex); //123123
    	}
    	else {
    		trainingindex++;
    		trainlabels.push_back(trainingindex); //1231
    	}
    }

    vector<Mat> testimages;
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
    vector<int> testlabels;

    int testingindex = 0;
    for (int i=0; i<testlabels_lfw.size(); i++) {
    	string currentlabel = testlabels_lfw[i];

    	if (currentlabel == testlabels_lfw[i+1]) {
    		testlabels.push_back(testingindex); //123123
    	}
    	else {
    		testingindex++;
    		testlabels.push_back(testingindex); //1231
    	}
    }

    // // // ------LFW Database--------// // //
	// // // ------ Perfrom LBP ------ // // //
	// // // --------------------------// // //
	Ptr<FaceRecognizer> face_model = createLBPHFaceRecognizer();
	face_model->train(trainimages_lfw, trainlabels_lfw_int);
	int pos = 0; int neg = 0; double errrate;
	for (int i=0; i<trainlabels_lfw_int.size(); i++) {
		int predictedLabel = face_model->predict(trainimages_lfw[i]);
		cout << "actual = " << trainlabels_lfw_int[i] << " predict = "<< predictedLabel<<endl;
		if (trainlabels_lfw_int[i] == predictedLabel) {
			pos++;
		}
		else {
			neg++;
		}
		errrate = (double) neg/trainlabels_lfw_int.size();
	}
	cout<< "LBP error rate = " << errrate <<endl << endl;
    Mat col1 = trainimages_lfw[0].reshape(0, trainimages_lfw[0].rows);
    Mat lbpres(trainimages_lfw[0].rows, trainimages_lfw[0].cols, CV_8UC1);
    olbp(col1, lbpres);
    imshow("Original image LFW", col1);
    imshow("After LBP operator", lbpres);
    imwrite("/Users/calvinlee/eigenfaces/output/Original_LFW.jpg"  ,col1);
    imwrite("/Users/calvinlee/eigenfaces/output/LBPresult_LFW.jpg"  ,lbpres);

 	// // // -------------------------------- // // //
    // // // ------ Perform Fisherface ------ // // //
	// // // ---------------------------------// // //
    Ptr<FaceRecognizer> fisher_model = createFisherFaceRecognizer();
    fisher_model->train(trainimages, trainlabels);
    int positive = 0; int negative = 0; double errorrate_fisher;
    for (int i=0; i<testimages.size(); i++) {
    	int predictedLabel_fisher = fisher_model->predict(testimages[i]);
    	cout << "actual = " << testlabels[i] << " fisherface predict = "<< predictedLabel_fisher<<endl;
    	if (testlabels[i] == predictedLabel_fisher) {
			positive++;
		}
		else {
			negative++;
		}
		errorrate_fisher = (double)negative/testlabels.size();
    }
    cout<< "Fisher error rate = "<<errorrate_fisher<<endl;


	// // // --------------------------------------------- // // //
	// // // ------ Show all 10 fisherfaces together ----- // // //
	// // // --------------------------------------------- // // //
    Mat W = fisher_model->getMat("eigenvectors");
    vector<Mat> fisherFaces;
    Mat all_fisherFacesMat;
    for (int i=0; i<min(10,W.cols); i++) {
    	Mat ev = W.col(i).clone();
    	Mat grayscale = toGrayscale(ev.reshape(1, trainimages[0].rows));
    	fisherFaces.push_back(grayscale);
    }
	hconcat(fisherFaces, all_fisherFacesMat);
	imwrite("/Users/calvinlee/eigenfaces/output/fisherfaces_all" + to_string(d) + ".jpg", all_fisherFacesMat);
	imshow("All 10 Fisherface" , all_fisherFacesMat);



    // // // ---------------------------------// // //
	// // // ------ Histogram Plotting ------ // // //
	// // // ---------------------------------// // //
    // int histSize = 256;    // bin size
    // float range[] = { 0, 255 };
    // const float *ranges[] = { range };

    // // Calculate histogram
    // MatND hist;
    // calcHist( &col1, 1, 0, Mat(), hist, 1, &histSize, ranges, true, false );

    // // Show the calculated histogram in command window
    // double total;
    // total = col1.rows * col1.cols;
    // for( int h = 0; h < histSize; h++ )
    //      {
    //         float binVal = hist.at<float>(h);
    //         cout<<" "<<binVal;
    //      }

    // // Plot the histogram
    // int hist_w = 512; int hist_h = 400;
    // int bin_w = cvRound( (double) hist_w/histSize );

    // Mat histImage( hist_h, hist_w, CV_8UC1, Scalar( 0,0,0) );
    // normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

    // for( int i = 1; i < histSize; i++ )
    // {
    //   line( histImage, Point( bin_w*(i-1), hist_h - cvRound(hist.at<float>(i-1)) ) ,
    //                    Point( bin_w*(i), hist_h - cvRound(hist.at<float>(i)) ),
    //                    Scalar( 255, 0, 0), 2, 8, 0  );
    // }
    // namedWindow( "Result", 1 );
    // imshow( "Before LBP Result", histImage );


    // // // ----------------------------------------------------------------------------------// // //
    // // // ------ Step 2: Trnasfrom the vectro into a matrix: Each image is a col vec ------ // // //
    // // // ----------------------------------------------------------------------------------// // //
    Mat trainMatrix = set2matrix(trainimages);
    Mat testMatrix = set2matrix(testimages);
    Mat meanVector(trainMatrix.rows, 1, CV_32FC1);
	reduce(trainMatrix, meanVector, 1, CV_REDUCE_AVG);
	imwrite("/Users/calvinlee/eigenfaces/output/MeanFace.jpg"  , meanVector.reshape(0, trainimages[0].rows));
	imshow("Mean Face",toGrayscale(meanVector.reshape(0, trainimages[0].rows)));
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
	Mat eigenFacesT = getEigenFace(eigenVector, demeanedFaces, d);	// d x 2500: Perform v_i=A*u_i
		vector<Mat> eigenFacesT_vec;
		vector<Mat> eigenFacesT_vec_normalized;
		Mat all_eigenFacesMat;	// display eigenfaces altogether
		for (int i = 0; i < d; i++){
			Mat eigenFacesT_normalized;
			eigenFacesT_vec.push_back((eigenFacesT.row(i)).reshape(0, trainimages[0].rows)); //put in the vector first
			eigenFacesT_normalized = toGrayscale(eigenFacesT_vec[i]);
			eigenFacesT_vec_normalized.push_back(eigenFacesT_normalized);
		}

	// // // ------ Show all 10 eigenfaces together ------ // // //
	hconcat(eigenFacesT_vec_normalized, all_eigenFacesMat);
	imwrite("/Users/calvinlee/eigenfaces/output/eigenfaces_all" + to_string(d) + ".jpg", all_eigenFacesMat);
	imshow("All " + to_string(d) + " eigenfaces" , all_eigenFacesMat);

	// // // ------ Normalize the eignfaces ------ // // //
	Mat eigenFacesNormT(d, trainMatrix.rows, CV_32FC1);
	for (int i = 0; i < eigenFacesT.cols; i++){	// Normalizar la transformación v_i = A * u_i
		normalize(eigenFacesT.col(i), eigenFacesNormT.col(i));
	}
	Mat eigenFacesNorm(trainMatrix.rows, d, CV_32FC1);
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

		// !classify 1 image! //
	// int choice = 0;
	// Mat testSample = projection1.col(choice); //projection of the testing image
	// for (int i=0; i<train_size; i++){
	// 	double dist = norm(testSample, groundProj.col(i), NORM_L2);
	// 	if (dist < minDist){
	// 		minDist = dist;
	// 		my_index = i;
	// 	}
	// }
	// cout << "actual = " << testlabels[choice] << +", Predicted = "<<labels[my_index]<<endl;

	cout << "Press CTRL-C or ENTER to exit program." << endl;
	cvWaitKey(0);
	return 0;
}