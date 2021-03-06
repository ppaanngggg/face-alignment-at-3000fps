#ifndef UTILS_H
#define UTILS_H
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <fstream>
#include "liblinear/linear.h"
#include <stdio.h>
#include <sys/time.h>
#include <thread>
#include <atomic>

class BoundingBox {
	public:
		double start_x;
		double start_y;
		double width;
		double height;
		double center_x;
		double center_y;
		BoundingBox(){
			start_x = 0;
			start_y = 0;
			width = 0;
			height = 0;
			center_x = 0;
			center_y = 0;
		}
};

class FeatureLocations
{
	public:
		cv::Point2d start;
		cv::Point2d end;
		FeatureLocations(cv::Point2d a, cv::Point2d b){
			start = a;
			end = b;
		}
		FeatureLocations(){
			start = cv::Point2d(0.0, 0.0);
			end = cv::Point2d(0.0, 0.0);
		};
};

class Parameters {
	//private:
	public:
		int local_features_num_;
		int landmarks_num_per_face_;
		int regressor_stages_;
		int tree_depth_;
		int trees_num_per_forest_;
		std::vector<double> local_radius_by_stage_;
		int initial_guess_;
		cv::Mat_<double> mean_shape_;
};

cv::Mat_<double> ProjectShape(const cv::Mat_<double>& shape, const BoundingBox& bbox);
cv::Mat_<double> ReProjection(const cv::Mat_<double>& shape, const BoundingBox& bbox);
cv::Mat_<double> GetMeanShape(const std::vector<cv::Mat_<double> >& all_shapes,
		const std::vector<BoundingBox>& all_bboxes);
void getSimilarityTransform(const cv::Mat_<double>& shape_to,
		const cv::Mat_<double>& shape_from,
		cv::Mat_<double>& rotation, double& scale
		);
void getAffineTransform(const cv::Mat_<double>& shape_to,
		const cv::Mat_<double>& shape_from,
		cv::Mat_<double>& affine
		);
cv::Mat_<double> doAffineTransform(
		const cv::Mat_<double>& shape,
		const cv::Mat_<double>& affine
		);

//cv::Mat_<double> LoadGroundTruthShape(std::string& name);
cv::Mat_<double> LoadGroundTruthShape(const char* name);

void LoadImages(
		std::vector<cv::Mat_<uchar> >& images, 
		std::vector<cv::Mat_<double> >& ground_truth_shapes,
		std::vector<BoundingBox>& bboxes, 
		std::string file_names,
		std::string set_name);

double CalculateError(cv::Mat_<double>& ground_truth_shape, cv::Mat_<double>& predicted_shape);

void DrawPredictImage(const cv::Mat_<uchar>& image, const cv::Mat_<double>& shape);

BoundingBox GetBoundingBox(cv::Mat_<double>& shape);

int ComputePixelDifferenct(
                           const FeatureLocations &pos,
                           const cv::Mat_<uchar> &image,
                           const cv::Mat_<double> &shape,
                           const BoundingBox &bbox,
                           const int landmark,
                           const cv::Mat_<double> &affine
                           );

#endif
