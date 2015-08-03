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

bool ShapeInRect(cv::Mat_<double>& shape, cv::Rect& ret){
	double sum_x = 0.0, sum_y = 0.0;
	double max_x = 0, min_x = 10000, max_y = 0, min_y = 10000;
	for (int i = 0; i < shape.rows; i++){
		if (shape(i, 0)>max_x) max_x = shape(i, 0);
		if (shape(i, 0)<min_x) min_x = shape(i, 0);
		if (shape(i, 1)>max_y) max_y = shape(i, 1);
		if (shape(i, 1)<min_y) min_y = shape(i, 1);

		sum_x += shape(i, 0);
		sum_y += shape(i, 1);
	}
	sum_x /= shape.rows;
	sum_y /= shape.rows;

	if ((max_x - min_x) > ret.width * 1.5) return false;
	if ((max_y - min_y) > ret.height * 1.5) return false;
    if (std::abs(sum_x - (ret.x + ret.width / 2.0)) > ret.width / 2.0) return false;
    if (std::abs(sum_y - (ret.y + ret.height / 2.0)) > ret.height / 2.0) return false;
	return true;
}

std::vector<cv::Rect> DetectFaces(cv::Mat_<uchar>& image, cv::CascadeClassifier& classifier){
	std::vector<cv::Rect_<int> > faces;
	classifier.detectMultiScale(image, faces, 1.1, 2, 0, cv::Size(30, 30));
	return faces;
}


void LoadImages(std::vector<cv::Mat_<uchar> >& images,
	std::vector<cv::Mat_<double> >& ground_truth_shapes,
	std::vector<BoundingBox>& bboxes,
	std::string file_names)
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

        cv::Mat_<uchar> image = cv::imread(("./../helen/testset/" + name + ".jpg").c_str(), 0);
        cv::Mat_<double> ground_truth_shape = LoadGroundTruthShape(("./../helen/testset/" + name + ".pts").c_str());
        BoundingBox bbox = GetBoundingBox(ground_truth_shape);
        images.push_back(image);
        ground_truth_shapes.push_back(ground_truth_shape);
        bboxes.push_back(bbox);

    //    cv::rectangle(image, cv::Point(bbox.start_x, bbox.start_y), cv::Point(bbox.start_x + bbox.width, bbox.start_y + bbox.height), cv::Scalar(255, 0, 0));
    //    DrawPredictImage(image, ground_truth_shape);

        count++;
        if (count%100 == 0){
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

void DrawPredictImage(const cv::Mat_<uchar> &image, cv::Mat_<double>& shape){
    cv::Mat tmp_img;
    cv::cvtColor(image, tmp_img, cv::COLOR_GRAY2BGR);
	for (int i = 0; i < shape.rows; i++){
        cv::circle(tmp_img, cv::Point2f(shape(i, 0), shape(i, 1)), 2, cv::Scalar(255,0,0));
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
