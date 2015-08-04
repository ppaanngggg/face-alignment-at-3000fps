#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <time.h>
#include "headers.h"
#include <sys/types.h>
#include <sys/stat.h>
//#include <omp.h>

using namespace cv;
using namespace std;

void Test(const char* ModelName)
{
	CascadeRegressor cas_load;
	cas_load.LoadCascadeRegressor(ModelName);
	std::vector<cv::Mat_<uchar> > images;
	std::vector<cv::Mat_<double> > ground_truth_shapes;
	std::vector<BoundingBox> bboxes;
	std::string file_names = "./../helen/testset/1.txt";
	LoadImages(images, ground_truth_shapes, bboxes, file_names, "testset/");
	// for (size_t i = 0; i < images.size(); i++) {
	// 	cv::imshow("image", images[i]);
	// 	cv::waitKey();
	// }
	double error;
	struct timeval t1, t2;
	gettimeofday(&t1, NULL);
	for (int i = 0; i < images.size(); i++){
		cv::Mat_<double> current_shape = ReProjection(cas_load.params_.mean_shape_, bboxes[i]);
		//gettimeofday(&t1, NULL);
		cv::Mat_<double> res = cas_load.Predict(images[i], current_shape, bboxes[i]);//, ground_truth_shapes[i]);

		// DrawPredictImage(images[i], res);
		error += CalculateError(ground_truth_shapes[i], res);
	}
	gettimeofday(&t2, NULL);
	double time_full = t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0;
	cout << "time full: " << time_full << " : " << time_full/images.size() << endl;
	cout << "error : " << error / images.size() << endl;
	return;
}

void Test(const char* ModelName, const char* picName)
{
	CascadeRegressor cas_load;
	cas_load.LoadCascadeRegressor(ModelName);

	cv::Mat_<uchar> image = cv::imread(picName, 0);

	std::string fn_haar = "./../haarcascade_frontalface_alt2.xml";
	cv::CascadeClassifier haar_cascade;
	bool yes = haar_cascade.load(fn_haar);
	std::cout << "detector: " << yes << std::endl;

	std::vector<cv::Rect> faces;
	haar_cascade.detectMultiScale(image, faces, 1.1, 2, 0, cv::Size(30, 30));

	for (int i = 0; i < faces.size(); i++) {
		cv::Rect faceRec = faces[i];
        BoundingBox bbox;
        bbox.start_x = faceRec.x;
        bbox.start_y = faceRec.y;
        bbox.width = faceRec.width;
        bbox.height = faceRec.height;
        bbox.center_x = bbox.start_x + bbox.width / 2.0;
        bbox.center_y = bbox.start_y + bbox.height / 2.0;

		cv::Mat_<double> current_shape = ReProjection(cas_load.params_.mean_shape_, bbox);
        cv::Mat_<double> res = cas_load.Predict(image, current_shape, bbox);
		DrawPredictImage(image, res);
	}
}

void Train(const char* ModelName)
{
	std::vector<cv::Mat_<uchar> > images;
	std::vector<cv::Mat_<double> > ground_truth_shapes;
	std::vector<BoundingBox> bboxes;
	std::string file_names = "./../helen/trainset/1.txt";
	LoadImages(images, ground_truth_shapes, bboxes, file_names, "trainset/");

	Parameters params;
	params.local_features_num_ = 100;
	params.landmarks_num_per_face_ = 68;
	params.regressor_stages_ = 4;
	params.local_radius_by_stage_.push_back(0.4);
	params.local_radius_by_stage_.push_back(0.3);
	params.local_radius_by_stage_.push_back(0.2);
	params.local_radius_by_stage_.push_back(0.1);
	//params.local_radius_by_stage_.push_back(0.08);
	params.local_radius_by_stage_.push_back(0.05);
	params.tree_depth_ = 3;
	params.trees_num_per_forest_ = 4;
	params.initial_guess_ = 1;

	params.mean_shape_ = GetMeanShape(ground_truth_shapes, bboxes);
	CascadeRegressor cas_reg;
	cas_reg.Train(images, ground_truth_shapes, bboxes, params);

	cas_reg.SaveCascadeRegressor(ModelName);
	return;
}

int main(int argc, char* argv[])
{
	if (argc >= 3)
	{
		if (strcmp(argv[1], "train") == 0)
		{
			std::cout << "enter train\n";
			Train(argv[2]);

			return 0;
		}
		if (strcmp(argv[1], "test") == 0)
		{
			std::cout << "enter test\n";
			if (argc == 3){
				Test(argv[2]);
			} else {
				Test(argv[2], argv[3]);
			}
			return 0;
		}
	}

	std::cout << "use [./application train ModelName] or [./application test ModelName [image_name]] \n";
	return 0;
}
