#include "utils.h"
//#include "facedetect-dll.h"
//#pragma comment(lib,"libfacedetect.lib")

// project the global shape coordinates to [-1, 1]x[-1, 1]
cv::Mat_<double> ProjectShape(const cv::Mat_<double>& shape, const BoundingBox& bbox)
{
	cv::Mat_<double> results(shape.rows, 2);

	// for (int i = 0; i < shape.rows; i++){
	// 	results(i, 0) = (shape(i, 0) - bbox.center_x) / (bbox.width / 2.0);
	// 	results(i, 1) = (shape(i, 1) - bbox.center_y) / (bbox.height / 2.0);
	// }
	double mean_x = cv::mean(shape.col(0))[0];
	double mean_y = cv::mean(shape.col(1))[0];
	cv::Mat((shape.col(0) - mean_x) / bbox.width).copyTo(results.col(0));
	cv::Mat((shape.col(1) - mean_y) / bbox.height).copyTo(results.col(1));

	return results;
}

// reproject the shape to global coordinates
cv::Mat_<double> ReProjection(const cv::Mat_<double>& shape, const BoundingBox& bbox)
{
	cv::Mat_<double> results(shape.rows, 2);
	// for (int i = 0; i < shape.rows; i++){
	// 	results(i, 0) = shape(i, 0)*bbox.width / 2.0 + bbox.center_x;
	// 	results(i, 1) = shape(i, 1)*bbox.height / 2.0 + bbox.center_y;
	// }
	double left_x, right_x;
	cv::minMaxIdx(shape.col(0), &left_x, &right_x);
	double top_y, bottom_y;
	cv::minMaxIdx(shape.col(1), &top_y, &bottom_y);

	double width_union = right_x - left_x;
	double height_union = bottom_y - top_y;

	cv::Mat(shape.col(0) - left_x).copyTo(results.col(0));
	cv::Mat(shape.col(1) - top_y).copyTo(results.col(1));

	cv::Mat(results.col(0) / width_union * bbox.width + bbox.start_x).copyTo(results.col(0));
	cv::Mat(results.col(1) / height_union * bbox.height + bbox.start_y).copyTo(results.col(1));

	return results;
}

// get the mean shape, [-1, 1]x[-1, 1]
cv::Mat_<double> GetMeanShape(const std::vector<cv::Mat_<double> >& all_shapes,
		const std::vector<BoundingBox>& all_bboxes)
{
	cv::Mat_<double> mean_shape = cv::Mat::zeros(all_shapes[0].rows, 2, CV_64F);
	for (int i = 0; i < all_shapes.size(); i++)
	{
		mean_shape += ProjectShape(all_shapes[i], all_bboxes[i]);
	}
	mean_shape = 1.0 / all_shapes.size()*mean_shape;
	// std::cout<<mean_shape<<std::endl;
	// exit(0);
	return mean_shape;
}

void getAffineTransform(const cv::Mat_<double>& shape_to,
		const cv::Mat_<double>& shape_from,
		cv::Mat_<double>& affine)
{
	cv::Mat_<double> shape_from_homo = cv::Mat::ones(shape_from.rows, shape_from.cols + 1, shape_from.depth());
	shape_from.copyTo(shape_from_homo.colRange(0,2));

	cv::Mat dst_x;
	cv::Mat dst_y;
	cv::solve(shape_from_homo, shape_to.col(0), dst_x, cv::DECOMP_SVD);
	cv::solve(shape_from_homo, shape_to.col(1), dst_y, cv::DECOMP_SVD);

	// std::cout<<"shape_from_homo"<<shape_from_homo<<std::endl;
	// std::cout<<"shape_to"<<shape_to<<std::endl;
	// std::cout<<"dst_x"<<dst_x<<std::endl;
	// std::cout<<"dst_y"<<dst_y<<std::endl;

	affine = cv::Mat::zeros(3, 2, affine.depth());
	dst_x.copyTo(affine.col(0));
	dst_y.copyTo(affine.col(1));
	// std::cout<<"affine"<<affine<<std::endl;
}

cv::Mat_<double> doAffineTransform(const cv::Mat_<double>& shape, const cv::Mat_<double>& affine)
{
	cv::Mat_<double> tmp = cv::Mat::zeros(shape.rows, 3, shape.depth());
	shape.copyTo(tmp.colRange(0,2));
	// cv::Mat_<double> ret = shape * affine.rowRange(0, 2);
	// cv::Mat(ret.col(0) - affine(2, 0)).copyTo(ret.col(0));
	// cv::Mat(ret.col(1) - affine(2, 1)).copyTo(ret.col(1));
	return tmp * affine;
}

// get the rotation and scale parameters by transferring shape_from to shape_to, shape_to = M*shape_from
void getSimilarityTransform(const cv::Mat_<double>& shape_to,
		const cv::Mat_<double>& shape_from,
		cv::Mat_<double>& rotation, double& scale){
	rotation = cv::Mat(2, 2, 0.0);
	scale = 0;

	// center the data
	double center_x_1 = cv::mean(shape_to.col(0))[0];
	double center_y_1 = cv::mean(shape_to.col(1))[0];
	double center_x_2 = cv::mean(shape_from.col(0))[0];
	double center_y_2 = cv::mean(shape_from.col(1))[0];

	cv::Mat_<double> temp1 = shape_to.clone();
	cv::Mat_<double> temp2 = shape_from.clone();

	temp1.col(0) -= center_x_1;
	temp1.col(1) -= center_y_1;
	temp2.col(0) -= center_x_2;
	temp2.col(1) -= center_y_2;


	cv::Mat_<double> covariance1, covariance2;
	cv::Mat_<double> mean1, mean2;
	// calculate covariance matrix
	cv::calcCovarMatrix(temp1, covariance1, mean1, cv::COVAR_COLS, CV_64F); //CV_COVAR_COLS
	cv::calcCovarMatrix(temp2, covariance2, mean2, cv::COVAR_COLS, CV_64F);

	double s1 = sqrt(norm(covariance1));
	double s2 = sqrt(norm(covariance2));
	scale = s1 / s2;
	temp1 = 1.0 / s1 * temp1;
	temp2 = 1.0 / s2 * temp2;

	double num = 0.0;
	double den = 0.0;
	for (int i = 0; i < shape_to.rows; i++){
		num = num + temp1(i, 1) * temp2(i, 0) - temp1(i, 0) * temp2(i, 1);
		den = den + temp1(i, 0) * temp2(i, 0) + temp1(i, 1) * temp2(i, 1);
	}

	double norm = sqrt(num*num + den*den);
	double sin_theta = num / norm;
	double cos_theta = den / norm;
	rotation(0, 0) = cos_theta;
	rotation(0, 1) = -sin_theta;
	rotation(1, 0) = sin_theta;
	rotation(1, 1) = cos_theta;
}

cv::Mat_<double> LoadGroundTruthShape(const char* name){
	int landmarks = 0;
	std::ifstream fin;
	std::string temp;
	fin.open(name, std::fstream::in);
	getline(fin, temp);// read first line
	fin >> temp >> landmarks;
	cv::Mat_<double> shape(landmarks, 2);
	getline(fin, temp); // read '\n' of the second line
	getline(fin, temp); // read third line
	for (int i = 0; i<landmarks; i++){
		fin >> shape(i, 0) >> shape(i, 1);
		//        std::cout<<shape(i, 0)<<" "<<shape(i, 1)<<std::endl;
	}
	fin.close();
	return shape;
}

void LoadImages(std::vector<cv::Mat_<uchar> >& images,
		std::vector<cv::Mat_<double> >& ground_truth_shapes,
		std::vector<BoundingBox>& bboxes,
		std::string file_names,
		std::string set_name)
{

	std::cout << "loading images\n";
	std::ifstream fin;
	fin.open(file_names.c_str(), std::ifstream::in);
	std::string name;
	int count = 0;
	//std::cout << name << std::endl;
	while (fin >> name){
		//std::cout << "reading file: " << name << std::endl;
		//std::cout << name << std::endl;

		cv::Mat_<uchar> image = cv::imread(("./../helen/" + set_name + name + ".jpg").c_str(), 0);
		cv::Mat_<double> ground_truth_shape = LoadGroundTruthShape(("./../helen/" + set_name + name + ".pts").c_str());
		BoundingBox bbox = GetBoundingBox(ground_truth_shape);
		images.push_back(image);
		ground_truth_shapes.push_back(ground_truth_shape);
		bboxes.push_back(bbox);

		cv::Mat_<uchar> image_flip;
		cv::flip(image, image_flip, 1);
		cv::Mat_<double> ground_truth_shape_flip(ground_truth_shape.rows, ground_truth_shape.cols);
//        std::cout<<ground_truth_shape_flip.rows<<" "<<ground_truth_shape_flip.cols<<std::endl;
		cv::Mat(image_flip.cols - ground_truth_shape.col(0)).copyTo(ground_truth_shape_flip.col(0));
		ground_truth_shape.col(1).copyTo(ground_truth_shape_flip.col(1));
		BoundingBox bbox_flip = GetBoundingBox(ground_truth_shape_flip);
		images.push_back(image_flip);
		ground_truth_shapes.push_back(ground_truth_shape_flip);
		bboxes.push_back(bbox_flip);

		//    cv::rectangle(image, cv::Point(bbox.start_x, bbox.start_y), cv::Point(bbox.start_x + bbox.width, bbox.start_y + bbox.height), cv::Scalar(255, 0, 0));
		   DrawPredictImage(image_flip, ground_truth_shape_flip);

		count++;
		if (count%200 == 0){
			std::cout << count << " images loaded\n";
            return;
		}
	}
	std::cout << "get " << bboxes.size() << " faces\n";
	fin.close();
}


double CalculateError(cv::Mat_<double>& ground_truth_shape, cv::Mat_<double>& predicted_shape){
	cv::Mat_<double> temp;
	double sum = 0;
	for (int i = 0; i<ground_truth_shape.rows; i++){
		sum += cv::norm(ground_truth_shape.row(i) - predicted_shape.row(i));
	}
	return sum / (ground_truth_shape.rows);
}

void DrawPredictImage(const cv::Mat_<uchar> &image, const cv::Mat_<double>& shape){
	cv::Mat tmp_img;
	cv::Mat_<double> tmp_shape = shape.clone();
	cv::cvtColor(image, tmp_img, cv::COLOR_GRAY2BGR);
	while (tmp_img.rows > 1024) {
		cv::resize(tmp_img, tmp_img, cv::Size(tmp_img.cols / 2, tmp_img.rows / 2));
		tmp_shape /= 2;
	}
	for (int i = 0; i < tmp_shape.rows; i++){
		cv::circle(tmp_img, cv::Point2f(tmp_shape(i, 0), tmp_shape(i, 1)), 2, cv::Scalar(255,0,0));
	}
	cv::imshow("show image", tmp_img);
	cv::waitKey(0);
}

BoundingBox GetBoundingBox(cv::Mat_<double>& shape)
{
	BoundingBox bbox;
	double left_x;
	double right_x;
	cv::minMaxIdx(shape.col(0), &left_x, &right_x);
	double top_y;
	double bottom_y;
	cv::minMaxIdx(shape.col(1), &top_y, &bottom_y);

	bbox.start_x = left_x;
	bbox.start_y = top_y;
	bbox.width = right_x - left_x + 1;
	bbox.height = bottom_y - top_y + 1;
	bbox.center_x = bbox.start_x + bbox.width / 2.;
	bbox.center_y = bbox.start_y + bbox.height / 2.;

	return bbox;
}

int ComputePixelDifferenct(
                           const FeatureLocations &pos,
                           const cv::Mat_<uchar> &image,
                           const cv::Mat_<double> &shape,
                           const BoundingBox &bbox,
                           const int landmark,
                           const cv::Mat_<double> &affine
                           )
{
    //get first point's pixel
    double delta_x = pos.start.x;
    double delta_y = pos.start.y;
    delta_x *= bbox.width;
    delta_y *= bbox.height;
    delta_x = affine(0, 0)*delta_x + affine(1, 0)*delta_y + affine(2, 0);
    delta_y = affine(0, 1)*delta_x + affine(1, 1)*delta_y + affine(2, 1);
    int real_x = delta_x + shape(landmark, 0);
    int real_y = delta_y + shape(landmark, 1);
    real_x = std::max(0, std::min(real_x, image.cols - 1)); // which cols
    real_y = std::max(0, std::min(real_y, image.rows - 1)); // which rows
    int tmp = (int)image(real_y, real_x); //real_y at first

    //           cv::circle(tmp_image, cv::Point2f(real_x, real_y), 2, cv::Scalar(0 ,0,255));
    //get second point's pixel
    delta_x = pos.end.x;
    delta_y = pos.end.y;
    delta_x *= bbox.width;
    delta_y *= bbox.height;
    delta_x = affine(0, 0)*delta_x + affine(1, 0)*delta_y + affine(2, 0);
    delta_y = affine(0, 1)*delta_x + affine(1, 1)*delta_y + affine(2, 1);
    real_x = delta_x + shape(landmark, 0);
    real_y = delta_y + shape(landmark, 1);
    real_x = std::max(0, std::min(real_x, image.cols - 1)); // which cols
    real_y = std::max(0, std::min(real_y, image.rows - 1)); // which rows

    return tmp - (int)image(real_y, real_x);
}
