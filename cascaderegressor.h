#ifndef CASCADE_REGRESSOR_H
#define CASCADE_REGRESSOR_H

#include "./utils.h"
#include "./regressor.h"

class CascadeRegressor {
public:
	//model params
	Parameters params_;
	//gray images, gt shapes, gt bboxes
	std::vector<cv::Mat_<uchar> > images_;
	std::vector<cv::Mat_<double> > ground_truth_shapes_;
	std::vector<BoundingBox> bboxes_;
	//regressors for each stage
	std::vector<Regressor> regressors_;
public:
	CascadeRegressor();
	void Train(
		const std::vector<cv::Mat_<uchar> >& images,
		const std::vector<cv::Mat_<double> >& ground_truth_shapes,
		const std::vector<BoundingBox>& bboxes,
		Parameters& params
	);
	cv::Mat_<double> Predict(
		cv::Mat_<uchar>& image,
		cv::Mat_<double>& current_shape,
		BoundingBox& bbox
	);
	void LoadCascadeRegressor(std::string ModelName);
	void SaveCascadeRegressor(std::string ModelName);

};

#endif
