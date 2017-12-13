#include <iostream>
#include "my_refind/config.h"

//像素坐标系和相机坐标系（归一化的，就是说d=1）的转化
Point2f pixel2cam(const Point2d& p, const Mat& K);
Point2d cam2pixel(const Point2d& p, const Mat& K_right);

int main(int argc, char **argv) {
	if (argc != 5) {
		cerr
				<< "usage: ./area_reprojected left_pic right_pic stereo_camera_config target_detect_config"
				<< endl;
		return -1;
	}

	//读入左右两幅图片
	Mat img_left, img_right;
	img_left = imread(argv[1]);
	img_right = imread(argv[2]);
	cout << "read img ok" << endl;
	//读入相机参数
	Config stereo(argv[3]);
	cout << "read stereo ok" << endl;
	//对左右图片进行矫正
	Mat img_left_rect, img_right_rect;
	stereo.doRectifyL(img_left, img_left_rect);
	stereo.doRectifyR(img_right, img_right_rect);
	cout << "img calibrate ok" << endl;

	//对彩色图片灰度化，并作直方图均衡化
	Mat gray_left, gray_right;

	cvtColor(img_left_rect, gray_left, COLOR_BGR2GRAY);
	equalizeHist(gray_left, gray_left);
	cvtColor(img_right_rect, gray_right, COLOR_BGR2GRAY);
	equalizeHist(gray_right, gray_right);

	//初始化级联分类器，这里以opencv自带的人脸识别的xml文件为例
	CascadeClassifier cascade;
	cascade.load(argv[4]);

	//虽然识别的是多目标，但是我们只考虑单目标的，在灰度图中，找到target
	vector < Rect > target_left;
	cascade.detectMultiScale(gray_left, target_left, 1.1, 2,
			0 | CASCADE_SCALE_IMAGE, Size(30, 30));
	vector < Rect > target_right;
	cascade.detectMultiScale(gray_right, target_right, 1.1, 2,
			0 | CASCADE_SCALE_IMAGE, Size(30, 30));

	if (target_left.size() == 0 || target_right.size() == 0) {
		cout << "no target" << endl;
		return -1;
	}
	//初始化特征点和描述子，选择ORB特征，很快并且比较准
	vector<KeyPoint> keypoints_left, keypoints_right;

	Mat desciptors_left, descriptors_right;

	Ptr < ORB > orb = ORB::create();

	//提取target所在区域的特征点
	orb->detect(img_left_rect(target_left[0]), keypoints_left);
	orb->detect(img_right_rect(target_right[0]), keypoints_right);

	//讲特征点坐标值加上目标框的坐标值，回归到整个图像的坐标
	for (size_t i = 0; i < keypoints_left.size(); i++) {
		keypoints_left[i].pt = keypoints_left[i].pt
				+ Point2f(target_left[0].x, target_left[0].y);
	}
	for (size_t i = 0; i < keypoints_right.size(); i++) {
		keypoints_right[i].pt = keypoints_right[i].pt
				+ Point2f(target_right[0].x, target_right[0].y);
	}

	//在图像中计算描述子
	orb->compute(img_left_rect, keypoints_left, desciptors_left);
	orb->compute(img_right_rect, keypoints_right, descriptors_right);

	//匹配计算，用汉明码暴力匹配，通过描述子计算匹配对
	vector < DMatch > matches;
	BFMatcher matcher(NORM_HAMMING);
	matcher.match(desciptors_left, descriptors_right, matches);

	double max_dist = 0;
	for (size_t i = 0; i < desciptors_left.rows; i++) {
		double dist = matches[i].distance;
		if (dist > max_dist)
			max_dist = dist;
	}
	//选取好的匹配
	vector < DMatch > good_matches;
	for (size_t i = 0; i < desciptors_left.rows; i++) {
		if (matches[i].distance <= 0.5 * max_dist)
			good_matches.push_back(matches[i]);
	}

	cout << "Match ok" << endl;

	Mat matches_show;
	drawMatches(img_left_rect, keypoints_left, img_right_rect, keypoints_right,
			good_matches, matches_show);
	imshow("show_matches", matches_show);

	cout << "show Match" << endl;

	Mat T1 = (Mat_<double>(3, 4) << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0);
	Mat T2 =
			(Mat_<double>(3, 4) << stereo.L2R_R.at<double>(0, 0), stereo.L2R_R.at<
					double>(0, 1), stereo.L2R_R.at<double>(0, 2), stereo.L2R_T.at<
					double>(0, 0), stereo.L2R_R.at<double>(1, 0), stereo.L2R_R.at<
					double>(1, 1), stereo.L2R_R.at<double>(1, 2), stereo.L2R_T.at<
					double>(1, 0), stereo.L2R_R.at<double>(2, 0), stereo.L2R_R.at<
					double>(2, 1), stereo.L2R_R.at<double>(2, 2), stereo.L2R_T.at<
					double>(2, 0));

	Mat R =
			(Mat_<double>(3, 3) << stereo.L2R_R.at<double>(0, 0), stereo.L2R_R.at<
					double>(0, 1), stereo.L2R_R.at<double>(0, 2), stereo.L2R_R.at<
					double>(1, 0), stereo.L2R_R.at<double>(1, 1), stereo.L2R_R.at<
					double>(1, 2), stereo.L2R_R.at<double>(2, 0), stereo.L2R_R.at<
					double>(2, 1), stereo.L2R_R.at<double>(2, 2));

	Mat T =
			(Mat_<double>(3, 1) << stereo.L2R_T.at<double>(0, 0), stereo.L2R_T.at<
					double>(1, 0), stereo.L2R_T.at<double>(2, 0));

	//对好的匹配，把像素坐标映射到相机坐标系
	vector<Point2d> pts_1, pts_2;
	for (DMatch m : good_matches) {
		pts_1.push_back(pixel2cam(keypoints_left[m.queryIdx].pt, stereo.K_l));
		pts_2.push_back(pixel2cam(keypoints_right[m.trainIdx].pt, stereo.K_r));
	}

	cout << "caculate xyz" << endl;

	//opencv里自带的算3d坐标的函数，计算匹配点对在左相机坐标系下的3d坐标（不是归一化的，是齐次坐标系）
	Mat pts_4d;
	//test/halo/201712-13
	cout << "T1:\n" << endl;
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 4; j++) {
			cout << T1.at<double>(i,j) << " ";
		}
		cout << "\n" << endl;
	}
	cout << "T2:\n" << endl;
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 4; j++) {
			cout << T2.at<double>(i,j)<< " ";
		}
		cout << "\n" << endl;
	}

	triangulatePoints(T1, T2, pts_1, pts_2, pts_4d);
	cout << "caculate ok" << endl;
	//齐次的转化为非其次的
	vector < Point3d > points;
	for (int i = 0; i < pts_4d.cols; i++) {
		Mat x = pts_4d.col(i);
		x /= x.at<double>(3, 0);
		Point3d p(x.at<double>(0, 0), x.at<double>(1, 0), x.at<double>(2, 0));
		points.push_back(p);
	}

	//对目标3d坐标，我们求和取平均
	double x_left = 0, y_left = 0, z_left = 0;

	cout << "camleft feature point camera coordinate:" << endl;
	for (int i = 0; i < points.size(); i++) {
		x_left += points[i].x;
		y_left += points[i].y;
		z_left += points[i].z;
		Point2d pts1_cam(points[i].x / points[i].z, points[i].y / points[i].z);

		//比较一下 左相机坐标（归一化的）和经过计算得到的左相机坐标（归一化的）区别
		cout << "point in left camera: " << pts_1[i] << endl;
		cout << "point projected from 3d" << pts1_cam << "  d=" << points[i].z
				<< endl;
	}
	x_left /= points.size();
	y_left /= points.size();
	z_left /= points.size();

	cout << "target 3d corrdinate:" << endl << x_left << "		" << y_left << "		"
			<< z_left << endl;

	cout << endl;

	/*
	 cout<<"camright feature point camera coordinate:"<<endl;
	 vector<Point3d> points_right;

	 for(int i=0;i<points.size();i++)
	 {
	 //对左相机坐标系中的坐标旋转平移后，得到其在右相机坐标系中的3d坐标（非齐次）
	 Mat x = R*(Mat_<double>(3,1)<<points[i].x,points[i].y,points[i].z)+T;
	 Point3d p(
	 x.at<double>(0,0),
	 x.at<double>(1,0),
	 x.at<double>(2,0)
	 );
	 points_right.push_back(p);
	 }



	 for(int i=0;i<points_right.size();i++)
	 {
	 //归一化的右相机坐标系下点坐标（xy）
	 Point2d pts2_cam(points_right[i].x/points_right[i].z,
	 points_right[i].y/points_right[i].z
	 );
	 //比较一下 右相机坐标（归一化的）和经过计算得到的右相机坐标（归一化的）区别
	 cout<<"point in right camera: "<<pts_2[i]<<endl;
	 cout<<"point projected from 3d"<<pts2_cam<<"  d="<<points_right[i].z<<endl;
	 }

	 */

	rectangle(img_left_rect, target_left[0], Scalar(255, 255, 0), 3, 3, 0);
	rectangle(img_right_rect, target_right[0], Scalar(255, 0, 255), 3, 3, 0);

	imshow("search_left", img_left_rect);
	imshow("search_right", img_right_rect);

	waitKey();

	std::cout << "Hello, world!" << std::endl;
	return 0;
}

Point2f pixel2cam(const Point2d& p, const Mat& K) {
	return Point2f((p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
			(p.y - K.at<double>(1, 2)) / K.at<double>(1, 1));
}

Point2d cam2pixel(const Point2d& p, const Mat& K_right) {
	return Point2d(p.x * K_right.at<double>(0, 0) + K_right.at<double>(0, 2),
			p.y * K_right.at<double>(1, 1) + K_right.at<double>(1, 2));
}
