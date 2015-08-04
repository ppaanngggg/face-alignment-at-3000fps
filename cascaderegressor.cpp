#include "cascaderegressor.h"

CascadeRegressor::CascadeRegressor(){

}

void CascadeRegressor::Train(const std::vector<cv::Mat_<uchar> >& images,
		const std::vector<cv::Mat_<double> >& ground_truth_shapes,
		const std::vector<BoundingBox>& bboxes,
		Parameters& params){

	std::cout << "Start training..." << std::endl;
	images_ = images;
	params_ = params;
	bboxes_ = bboxes;
	ground_truth_shapes_ = ground_truth_shapes;

	//real datas for training
	std::vector<int> augmented_images_index; // just index in images_
	std::vector<BoundingBox> augmented_bboxes;
	std::vector<cv::Mat_<double> > augmented_ground_truth_shapes;
	std::vector<cv::Mat_<double> > augmented_current_shapes;

	//init random_generator
	time_t current_time;
	current_time = time(0);
	cv::RNG random_generator(current_time);

	std::cout << "augment data sets" << std::endl;
	for (int i = 0; i < images_.size(); i++){
		for (int j = 0; j < params_.initial_guess_; j++)
		{
            double angle = double(random_generator) - 0.5;
            cv::Mat_<double> rotation(2, 2);
            rotation(0, 0) = std::cos(angle); rotation(0, 1) = std::sin(angle);
            rotation(1, 0) = - std::sin(angle); rotation(1, 1) = std::cos(angle);
            cv::Mat_<double> temp = params_.mean_shape_ * rotation;
//            std::cout << temp<<std::endl;
//            cv::Mat_<double> scale(2, 2);
//            scale(0, 0) = double(random_generator) * 0.2 + 0.9; scale(0, 1) = 0;
//            scale(1, 0) = 0; scale(1, 1) = double(random_generator) * 0.2 + 0.9;
////            std::cout << temp<<std::endl;
//            temp *= scale;
//            std::cout << temp<<std::endl;
            
            
			// //choose init shape randomly
			// int index = 0;
			// do {
			// 	index = random_generator.uniform(0, images_.size());
			// }while(index == i);
			//
			// cv::Mat_<double> temp = ground_truth_shapes_[index];
			// temp = ProjectShape(temp, bboxes_[index]);
            
            temp = ReProjection(temp, bboxes_[i]);
            augmented_images_index.push_back(i);
            augmented_ground_truth_shapes.push_back(ground_truth_shapes_[i]);
            augmented_bboxes.push_back(GetBoundingBox(temp));
            augmented_current_shapes.push_back(temp);
            // DrawPredictImage(images[i], temp);
		}
		//choose mean shape for init
		cv::Mat_<double> augmented_mean_shape = ReProjection(params_.mean_shape_, bboxes_[i]);
		augmented_images_index.push_back(i);
		augmented_ground_truth_shapes.push_back(ground_truth_shapes_[i]);
		augmented_bboxes.push_back(GetBoundingBox(augmented_mean_shape));
		augmented_current_shapes.push_back(augmented_mean_shape);
	}

	std::cout << "augmented size: " << augmented_current_shapes.size() << std::endl;

	//train regressors for each stage
	regressors_.resize(params_.regressor_stages_);
	for (int i = 0; i < params_.regressor_stages_; i++){
		std::cout << "training stage: " << i << " of " << params_.regressor_stages_ << std::endl;
		//train and return increaments for each training data
		std::vector<cv::Mat_<double> > shape_increaments = regressors_[i].Train(
			images_,
			augmented_images_index,
			augmented_ground_truth_shapes,
			augmented_bboxes,
			augmented_current_shapes,
			params_,
			i
		);
		std::cout << "update current shapes" << std::endl;
		double error = 0.0;
		for (int j = 0; j < shape_increaments.size(); j++){
			cv::Mat(shape_increaments[j].col(0) * augmented_bboxes[j].width).copyTo(shape_increaments[j].col(0));
			cv::Mat(shape_increaments[j].col(1) * augmented_bboxes[j].height).copyTo(shape_increaments[j].col(1));
			//update current shape and current bbox
			augmented_current_shapes[j] = shape_increaments[j] + augmented_current_shapes[j];
			augmented_bboxes[j] = GetBoundingBox(augmented_current_shapes[j]);
			error += CalculateError(augmented_ground_truth_shapes[j], augmented_current_shapes[j]);
		}

		std::cout << "regression error: " <<  error << ": " << error/shape_increaments.size() << std::endl;
	}
}


cv::Mat_<double> CascadeRegressor::Predict(cv::Mat_<uchar>& image,
		cv::Mat_<double>& current_shape, BoundingBox& bbox){

	for (int i = 0; i < params_.regressor_stages_; i++){
		bbox = GetBoundingBox(current_shape);

		cv::Mat mean_shape_resize = ReProjection(params_.mean_shape_, bbox);
		cv::Mat tmp_mean_shape_resize(params_.mean_shape_.rows, 2, params_.mean_shape_.depth());
		cv::Mat tmp_current_shape(params_.mean_shape_.rows, 2, params_.mean_shape_.depth());
		cv::Mat(mean_shape_resize.col(0) - mean(mean_shape_resize.col(0))).copyTo(tmp_mean_shape_resize.col(0));
		cv::Mat(mean_shape_resize.col(1) - mean(mean_shape_resize.col(1))).copyTo(tmp_mean_shape_resize.col(1));
		cv::Mat(current_shape.col(0) - mean(current_shape.col(0))).copyTo(
				tmp_current_shape.col(0));
		cv::Mat(current_shape.col(1) - mean(current_shape.col(1))).copyTo(
				tmp_current_shape.col(1));
		cv::Mat_<double> affine;
		getAffineTransform(tmp_mean_shape_resize, tmp_current_shape, affine);

		cv::Mat_<double> shape_increaments = regressors_[i].Predict(image, current_shape, bbox, affine);
		cv::Mat(shape_increaments.col(0) * bbox.width).copyTo(shape_increaments.col(0));
		cv::Mat(shape_increaments.col(1) * bbox.height).copyTo(shape_increaments.col(1));
		current_shape = shape_increaments + current_shape;
	}
	cv::Mat_<double> res = current_shape;
	return res;
}

void CascadeRegressor::LoadCascadeRegressor(std::string ModelName){
	std::ifstream fin;
	fin.open((ModelName + "_params.txt").c_str(), std::fstream::in);
	params_ = Parameters();
	fin >> params_.local_features_num_
		>> params_.landmarks_num_per_face_
		>> params_.regressor_stages_
		>> params_.tree_depth_
		>> params_.trees_num_per_forest_
		>> params_.initial_guess_;

	std::vector<double> local_radius_by_stage;
	local_radius_by_stage.resize(params_.regressor_stages_);
	for (int i = 0; i < params_.regressor_stages_; i++){
		fin >> local_radius_by_stage[i];
	}
	params_.local_radius_by_stage_ = local_radius_by_stage;

	cv::Mat_<double> mean_shape(params_.landmarks_num_per_face_, 2, 0.0);
	for (int i = 0; i < params_.landmarks_num_per_face_; i++){
		fin >> mean_shape(i, 0) >> mean_shape(i, 1);
	}
	params_.mean_shape_ = mean_shape;
	regressors_.resize(params_.regressor_stages_);
	for (int i = 0; i < params_.regressor_stages_; i++){
		regressors_[i].params_ = params_;
		regressors_[i].LoadRegressor(ModelName, i);
		regressors_[i].ConstructLeafCount();
	}
}


void CascadeRegressor::SaveCascadeRegressor(std::string ModelName){
	std::ofstream fout;
	fout.open((ModelName + "_params.txt").c_str(), std::fstream::out);
	fout << params_.local_features_num_ << " "
		<< params_.landmarks_num_per_face_ << " "
		<< params_.regressor_stages_ << " "
		<< params_.tree_depth_ << " "
		<< params_.trees_num_per_forest_ << " "
		<< params_.initial_guess_ << std::endl;
	for (int i = 0; i < params_.regressor_stages_; i++){
		fout << params_.local_radius_by_stage_[i] << std::endl;
	}
	for (int i = 0; i < params_.landmarks_num_per_face_; i++){
		fout << params_.mean_shape_(i, 0) << " " << params_.mean_shape_(i, 1) << std::endl;
	}

	fout.close();

	for (int i = 0; i < params_.regressor_stages_; i++){
		//regressors_[i].SaveRegressor(fout);
		regressors_[i].SaveRegressor(ModelName, i);
		//regressors_[i].params_ = params_;
	}

}
