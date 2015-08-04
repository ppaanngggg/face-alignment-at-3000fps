#include "regressor.h"
#include <time.h>
#include <assert.h>
//SYSTEM MACORS LISTS: http://sourceforge.net/p/predef/wiki/OperatingSystems/

#include <sys/types.h>
#include <sys/stat.h>

std::vector<cv::Mat_<double> > Regressor::Train(
	const std::vector<cv::Mat_<uchar> >& images,
	const std::vector<int>& augmented_images_index,
	const std::vector<cv::Mat_<double> >& augmented_ground_truth_shapes,
	const std::vector<BoundingBox>& augmented_bboxes,
	const std::vector<cv::Mat_<double> >& augmented_current_shapes,
	const Parameters& params,
	const int stage
)
{

	stage_ = stage;
	params_ = params;

	std::vector<cv::Mat_<double> > regression_targets;
	std::vector<cv::Mat_<double> > affines;
	regression_targets.resize(augmented_current_shapes.size());
	affines.resize(augmented_current_shapes.size());

	// calculate the regression targets
	std::cout << "calculate regression targets" << std::endl;
	#pragma omp parallel for
	for (int i = 0; i < augmented_current_shapes.size(); i++){
		//turn mean_shape and current_shape to their center, used to compute affine
		cv::Mat mean_shape_resize = ReProjection(params_.mean_shape_, augmented_bboxes[i]);
		cv::Mat tmp_mean_shape_resize(params.mean_shape_.rows, 2, params.mean_shape_.depth());	//alloc
		cv::Mat tmp_current_shape(params.mean_shape_.rows, 2, params.mean_shape_.depth());		//alloc
		cv::Mat(mean_shape_resize.col(0) - mean(mean_shape_resize.col(0))).copyTo(tmp_mean_shape_resize.col(0));
		cv::Mat(mean_shape_resize.col(1) - mean(mean_shape_resize.col(1))).copyTo(tmp_mean_shape_resize.col(1));
		cv::Mat(augmented_current_shapes[i].col(0) - mean(augmented_current_shapes[i].col(0))).copyTo(
				tmp_current_shape.col(0));
		cv::Mat(augmented_current_shapes[i].col(1) - mean(augmented_current_shapes[i].col(1))).copyTo(
				tmp_current_shape.col(1));
		// std::cout<<"tmp_mean_shape_resize :"<<tmp_mean_shape_resize<<std::endl;
		// std::cout<<"tmp_current_shape :"<<tmp_current_shape<<std::endl;

		//compute affine from current_shape to mean_shape
		cv::Mat_<double> affine;
		getAffineTransform(tmp_mean_shape_resize, tmp_current_shape, affine);

		//compute regression_target and transform it into union
		cv::Mat regression_target = augmented_ground_truth_shapes[i] - augmented_current_shapes[i];
		// std::cout<<"regression_target :"<<regression_target<<std::endl;
		regression_target = doAffineTransform(regression_target, affine);
		cv::Mat(regression_target.col(0) / augmented_bboxes[i].width).copyTo(regression_target.col(0));
		cv::Mat(regression_target.col(1) / augmented_bboxes[i].height).copyTo(regression_target.col(1));
		regression_targets[i] = regression_target;
		// std::cout<<"regression_target :"<<regression_target<<std::endl;
		// std::cout<<"affine matrix :"<<affine<<std::endl;

		//conpute affine from mean_shape to current_shape and add into vector
		getAffineTransform(tmp_current_shape, tmp_mean_shape_resize, affine);
		affines[i] = affine;
	}

	std::cout << "train forest of stage:" << stage_ << std::endl;
	rd_forests_.resize(params_.landmarks_num_per_face_);
	#pragma omp parallel for
	for (int i = 0; i < params_.landmarks_num_per_face_; ++i){
		std::cout << "\tlandmark: " << i << std::endl;
		rd_forests_[i] = RandomForest(params_, i, stage_, regression_targets);
		rd_forests_[i].TrainForest(
			images,
			augmented_images_index,
			augmented_bboxes,
			augmented_current_shapes,
			affines
		);
	}

	std::cout << "Get Global Binary Features" << std::endl;
	//build sparse feature_binary_code
	struct feature_node **global_binary_features;
	global_binary_features = new struct feature_node* [augmented_current_shapes.size()];
//	for(int i = 0; i < augmented_current_shapes.size(); ++i){
//		global_binary_features[i] = new feature_node[
//			params_.trees_num_per_forest_ * params_.landmarks_num_per_face_ + 1
//			];
//	}
	//compute feature_binary_code for each augmented data
	#pragma omp parallel for
	for (int i = 0; i < augmented_current_shapes.size(); ++i){
		//get info of [i]th augmented data
		const cv::Mat_<double>& affine = affines[i];
		const cv::Mat_<uchar>& image = images[augmented_images_index[i]];
		const BoundingBox& bbox = augmented_bboxes[i];
		const cv::Mat_<double>& current_shape = augmented_current_shapes[i];

        global_binary_features[i] = GetGlobalBinaryFeatures(image, current_shape, bbox, affine);
	}
	//count total num of features
	int num_feature = 0;
	for (int i=0; i < params_.landmarks_num_per_face_; ++i){
		num_feature += rd_forests_[i].all_leaf_nodes_;
		// std::cout<<"all_leaf_nodes :"<<rd_forests_[i].all_leaf_nodes_<<std::endl;
	}
	//set global regression params
	struct parameter* regression_params = new struct parameter;
	regression_params-> solver_type = L2R_L2LOSS_SVR_DUAL;
	regression_params->C = 1.0/augmented_current_shapes.size();
	regression_params->p = 0;

	std::cout << "Global Regression of stage " << stage_ << std::endl;
	// alloc model and targets
	linear_model_x_.resize(params_.landmarks_num_per_face_);
	linear_model_y_.resize(params_.landmarks_num_per_face_);
	double** targets = new double*[params_.landmarks_num_per_face_];
	for (int i = 0; i < params_.landmarks_num_per_face_; ++i){
		targets[i] = new double[augmented_current_shapes.size()];
	}
	#pragma omp parallel for
	for (int i = 0; i < params_.landmarks_num_per_face_; ++i){
		struct problem* prob = new struct problem;
		prob->l = augmented_current_shapes.size();
		prob->n = num_feature;
		prob->x = global_binary_features;
		prob->bias = -1;
		std::cout << "\tregress landmark " << i << std::endl;
		for(int j = 0; j< augmented_current_shapes.size();j++){
			targets[i][j] = regression_targets[j](i, 0);
		}
		prob->y = targets[i];
		check_parameter(prob, regression_params);
		struct model* regression_model = train(prob, regression_params);
		linear_model_x_[i] = regression_model;
		for(int j = 0; j < augmented_current_shapes.size(); j++){
			targets[i][j] = regression_targets[j](i, 1);
		}
		prob->y = targets[i];
		check_parameter(prob, regression_params);
		regression_model = train(prob, regression_params);
		linear_model_y_[i] = regression_model;
	}
	for (int i = 0; i < params_.landmarks_num_per_face_; ++i){
		delete[] targets[i];// = new double[augmented_current_shapes.size()];
	}
	delete[] targets;
	std::cout << "predict regression targets" << std::endl;

	std::vector<cv::Mat_<double> > predict_regression_targets;
	predict_regression_targets.resize(augmented_current_shapes.size());
	#pragma omp parallel for
	for (int i = 0; i < augmented_current_shapes.size(); i++){
		cv::Mat_<double> a(params_.landmarks_num_per_face_, 2, 0.0);
		for (int j = 0; j < params_.landmarks_num_per_face_; j++){
			a(j, 0) = predict(linear_model_x_[j], global_binary_features[i]);
			a(j, 1) = predict(linear_model_y_[j], global_binary_features[i]);
		}
		cv::Mat(a.col(0) * augmented_bboxes[i].width).copyTo(a.col(0));
		cv::Mat(a.col(1) * augmented_bboxes[i].height).copyTo(a.col(1));
		predict_regression_targets[i] = doAffineTransform(a, affines[i]);
		if (i%500 == 0 && i > 0){
			std::cout << "\tpredict " << i << " images" << std::endl;
		}
	}
	for (int i = 0; i< augmented_current_shapes.size(); i++){
		delete[] global_binary_features[i];
	}
	delete[] global_binary_features;

	return predict_regression_targets;
}

Regressor::Regressor(){
}

Regressor::Regressor(const Regressor &a){
}

Regressor::~Regressor(){

}

struct feature_node* Regressor::GetGlobalBinaryFeatures(const cv::Mat_<uchar>& image,
		const cv::Mat_<double>& current_shape, const BoundingBox& bbox, const cv::Mat_<double>& affine){
	int index = 1;

	struct feature_node* binary_features = new feature_node[params_.trees_num_per_forest_*params_.landmarks_num_per_face_+1];
	int ind = 0;
	for (int j = 0; j < params_.landmarks_num_per_face_; ++j)
	{
		for (int k = 0; k < params_.trees_num_per_forest_; ++k)
		{
			Node* node = rd_forests_[j].trees_[k];
			while (!node->is_leaf_){
				FeatureLocations& pos = node->feature_locations_;
                int pixel_difference = ComputePixelDifferenct(pos, image, current_shape, bbox, j, affine);
				if (pixel_difference < node->threshold_){
					node = node->left_child_;// go left
				}
				else{
					node = node->right_child_;// go right
				}
			}

			//int ind = j*params_.trees_num_per_forest_ + k;
			//int ind = feature_node_index[j] + k;
			//binary_features[ind].index = leaf_index_count[j] + node->leaf_identity;
			binary_features[ind].index = index + node->leaf_identity;
			binary_features[ind].value = 1.0;
			ind++;
			//std::cout << binary_features[ind].index << " ";
		}

		index += rd_forests_[j].all_leaf_nodes_;
	}
	//std::cout << "\n";
	//std::cout << index << ":" << params_.trees_num_per_forest_*params_.landmarks_num_per_face_ << std::endl;
	binary_features[params_.trees_num_per_forest_*params_.landmarks_num_per_face_].index = -1;
	binary_features[params_.trees_num_per_forest_*params_.landmarks_num_per_face_].value = -1.0;
	return binary_features;
}

cv::Mat_<double> Regressor::Predict(cv::Mat_<uchar>& image,
		cv::Mat_<double>& current_shape, BoundingBox& bbox, cv::Mat_<double>& affine){

	cv::Mat_<double> predict_result(current_shape.rows, current_shape.cols, 0.0);

	feature_node* binary_features = GetGlobalBinaryFeatures(image, current_shape, bbox, affine);

	for (int i = 0; i < current_shape.rows; i++){
		predict_result(i, 0) = predict(linear_model_x_[i], binary_features);
		predict_result(i, 1) = predict(linear_model_y_[i], binary_features);
	}

	delete[] binary_features;
	//delete[] tmp_binary_features;
	cv::Mat(predict_result.col(0) * bbox.width).copyTo(predict_result.col(0));
	cv::Mat(predict_result.col(1) * bbox.height).copyTo(predict_result.col(1));
	return doAffineTransform(predict_result, affine);
}

void Regressor::LoadRegressor(std::string ModelName, int stage){
	char buffer[50];
	sprintf(buffer, "%s_%d_regressor.txt", ModelName.c_str(), stage);
	std::ifstream fin;
	fin.open(buffer, std::fstream::in);
	int rd_size, linear_size;
	fin >> stage_ >> rd_size >> linear_size;
	rd_forests_.resize(rd_size);
	for (int i = 0; i < rd_size; i++){
		rd_forests_[i].LoadRandomForest(fin);
	}
	linear_model_x_.clear();
	linear_model_y_.clear();
	for (int i = 0; i < linear_size; i++){
		sprintf(buffer, "%s_%d/%d_linear_x.txt", ModelName.c_str(), stage_, i);
		linear_model_x_.push_back(load_model(buffer));

		sprintf(buffer, "%s_%d/%d_linear_y.txt", ModelName.c_str(), stage_, i);
		linear_model_y_.push_back(load_model(buffer));
	}
}

void Regressor::ConstructLeafCount(){
	int index = 1;
	int ind = params_.trees_num_per_forest_;
	for (int i = 0; i < params_.landmarks_num_per_face_; ++i){
		leaf_index_count[i] = index;
		index += rd_forests_[i].all_leaf_nodes_;
		feature_node_index[i] = ind*i;
	}
}

void Regressor::SaveRegressor(std::string ModelName, int stage){
	char buffer[50];
	//strcpy(buffer, ModelName.c_str());
	assert(stage == stage_);
	sprintf(buffer, "%s_%d_regressor.txt", ModelName.c_str(), stage);

	std::ofstream fout;
	fout.open(buffer, std::fstream::out);
	fout << stage_ << " "
		<< rd_forests_.size() << " "
		<< linear_model_x_.size() << std::endl;

	for (int i = 0; i < rd_forests_.size(); i++){
		rd_forests_[i].SaveRandomForest(fout);
	}

	for (
			int i = 0; i < linear_model_x_.size(); i++){
		sprintf(buffer, "%s_%d", ModelName.c_str(), stage_);

		struct stat st = {0};
		if (stat(buffer, &st) == -1) {
			mkdir(buffer, 0777);
		}

		//_mkdir(buffer);
		sprintf(buffer, "%s_%d/%d_linear_x.txt", ModelName.c_str(), stage_, i);
		save_model(buffer, linear_model_x_[i]);

		sprintf(buffer, "%s_%d/%d_linear_y.txt", ModelName.c_str(), stage_, i);
		save_model(buffer, linear_model_y_[i]);
	}
}
