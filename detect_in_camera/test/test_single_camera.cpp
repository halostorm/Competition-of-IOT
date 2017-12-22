#include <iostream>
#include <opencv2/opencv.hpp>
#include "my_refind/config.h"
#include <opencv2/core/utility.hpp>
#include <opencv2/tracking.hpp>

using namespace std;
using namespace cv;
//像素坐标系和相机坐标系（归一化的，就是说d=1）的转化
Point2f pixel2cam(const Point2d& p, const Mat& K);
Point2d cam2pixel(const Point2d& p, const Mat& K_right);

int main(int argc, char **argv) {
	if (argc != 2) {
		cerr
				<< "usage: ./area_reprojected  stereo_camera_config target_detect_config"
				<< endl;
		return -1;
	}
	cout << "program start ok" << endl;
	//init tracker//////////////////
	String tracker_algorithm = "KCF";
	//String video_name = parser.get<String>( 1 );
	Ptr<Tracker> tracker_left = Tracker::create(tracker_algorithm);
	if (tracker_left == NULL) {
		cout << "***Error in the instantiation of the left tracker...***\n";
		return -1;
	}

	//init camera,0-left/1-right
	VideoCapture cam_left(0);

	cout << "find camera ok" << endl;

	cam_left.set(CV_CAP_PROP_FRAME_WIDTH, 320);
	cam_left.set(CV_CAP_PROP_FRAME_HEIGHT, 240);

	//cam_left.set(CV_CAP_PROP_MODE, 1);
	//cam_right.set(CV_CAP_PROP_MODE, 1);

	cam_left.set(CV_CAP_PROP_FOURCC, CV_FOURCC('M', 'J', 'P', 'G'));

	cout << "camera set ok" << endl;

	//如果打开失败，报错
	if (!cam_left.isOpened()) {
		cout << "camera open error" << endl;
		return -1;

	}
	cout << "camera open ok" << endl;
	//init camera params

	//初始化级联分类器，这里以opencv自带的人脸识别的xml文件为例
	CascadeClassifier cascade;
	cascade.load(argv[1]);
	cout << "cascade load ok" << endl;
	//init values
	Mat pic_left;
	Mat gray_left;
	vector<Rect> target_left;
	//Rect2d target_left_box;
	bool left_detected = false;
	bool tracker_initialized = false;	//
	vector<KeyPoint> keypoints_left;
	Mat desciptors_left;

	Ptr<ORB> orb = ORB::create(500);

	int loop = 0;
	cout << "begin video" << endl;
	clock_t start, finish;
	start = clock();
	finish = clock();
	while (1) {
		//视频流输入图片
		cam_left >> pic_left;
		cout << "pic_time: " << finish - start << "/" << CLOCKS_PER_SEC
				<< " (s) " << endl;
		//多少个周期一检测，视情况调整
		if (loop % 10 == 0) {
			cout << "time: " << finish - start << "/" << CLOCKS_PER_SEC
					<< " (s) " << endl;
			start = clock();
			loop = 0;
			//先对图片做矫正
			if (!tracker_initialized) {
				//检测是否有目标
				//灰度化，均衡化
				//cvtColor(pic_left_rect, gray_left, COLOR_BGR2GRAY);
				//equalizeHist(gray_left, gray_left);
				//cvtColor(pic_right_rect, gray_right, COLOR_BGR2GRAY);
				//equalizeHist(gray_right, gray_right);

				cascade.detectMultiScale(pic_left, target_left, 1.1, 2,
						0 | CASCADE_SCALE_IMAGE, Size(30, 30));
			}
			if (target_left.size() == 1) {	//if left no target, continue
				left_detected = true;
			} else {
				left_detected = false;
				continue;
			}
			if (!tracker_initialized && left_detected) {//initializes the tracker//only do one time
				cout << "can init tracker" << endl;
				Rect2d target_left_box((double) target_left[0].x,
						(double) target_left[0].y,
						(double) target_left[0].width,
						(double) target_left[0].height);
				if (!tracker_left->init(pic_left, target_left_box)) {
					cout << "***Could not initialize left tracker...***\n";
					left_detected = false;
					continue;
				}
				tracker_initialized = true;
				continue;
			} else if (tracker_initialized) {
				//updates the tracker
				Rect2d target_left_box;
				Rect2d target_right_box;
				if (tracker_left->update(pic_left, target_left_box)) {
					rectangle(pic_left, target_left_box, Scalar(255, 0, 0), 2,
							1);
					left_detected = true;
				} else {
					left_detected = false;
				}

				//imshow( "Tracking API", image );
				//imshow("left",pic_left);
				//imshow("right",pic_right);

				//如果两边各有一个，认为是同一个
				if (left_detected) {
					cout << "one target, and detected by two camera" << endl;

					//提取target所在区域的特征点
					orb->detect(pic_left(target_left_box), keypoints_left);

					//将特征点坐标值加上目标框的坐标值，回归到整个图像的坐标
					for (size_t i = 0; i < keypoints_left.size(); i++) {
						keypoints_left[i].pt = keypoints_left[i].pt
								+ Point2f(target_left_box.x, target_left_box.y);
					}
					//在图像中计算描述子
					orb->compute(pic_left, keypoints_left, desciptors_left);
					imshow("left", pic_left);
					 cout<< "left detect and right not detect, use only left camera";
					 orb->detect(pic_left(target_left_box), keypoints_left);
					 for (size_t i = 0; i < keypoints_left.size(); i++) {
					 keypoints_left[i].pt = keypoints_left[i].pt
					 + Point2f(target_left_box.x, target_left_box.y);
					 }
					 vector<Point2d> pts_left;
					 for (auto k : keypoints_left) {
					 pts_left.push_back(pixel2cam(k.pt, stereo.K_l));
					 }
					 double x_left = 0, y_left = 0, z_left = 1000;
					 for (int i = 0; i < pts_left.size(); i++) {
					 x_left += pts_left[i].x;
					 y_left += pts_left[i].y;

					 }
					 x_left /= pts_left.size();
					 y_left /= pts_left.size();

					 cout << "target 3d corrdinate:" << endl << x_left * 1000
					 << "	" << y_left * 1000 << "	" << z_left << endl;

					imshow("left", pic_left);

				} else if (!left_detected) {//no left and no right
					cout << "no target，wait for next" << endl;
					imshow("left", pic_left);

				}
			}
			finish = clock();
		} else {
			loop++;
		}
		char c = waitKey(10);
		if (c == 32) {
			imwrite("../img_left.jpg", pic_left);
			imwrite("../img_right.jpg", pic_right);
		}
		if (c == 27) {
			break;
		}
		//imshow("left",pic_left);
		//imshow("right",pic_right);
	}
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
