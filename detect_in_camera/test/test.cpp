#include <iostream>
#include <opencv2/opencv.hpp>
#include "my_refind/config.h"
#include <opencv2/core/utility.hpp>
#include <opencv2/tracking.hpp>

#include <stdlib.h>
#include "my_refind/KFT.hpp"
#include "my_refind/LedController.h"
//#include "my_refind/motor.h"
#include "my_refind/para.h"
#include "my_refind/PID.h"
#include "my_refind/Predictor.hpp"
//#include "my_refind/uart.h"
#include <wiringPi.h>
#include <softPwm.h>

#define RANGE	200 //1 means 100 us , 200 means 20 ms 1等于100微妙，200等于20毫秒
#define X_PIN	1
#define Y_PIN	15

using namespace std;
using namespace cv;
//像素坐标系和相机坐标系（归一化的，就是说d=1）的转化
Point2f pixel2cam(const Point2d& p, const Mat& K);
Point2d cam2pixel(const Point2d& p, const Mat& K_right);
bool servo_init(int pin, int range);
bool servo_turn(int pin, double angle);

int main(int argc, char **argv) {
	if (argc != 3) {
		cerr
				<< "usage: ./test ../config/stereo_paras.yaml ../config/haarcascade_frontalface_alt.xml"
				<< endl;
		return -1;
	}
	cout << "program start ok" << endl;
	/*
	LedController led;
	for (int i = 0; i < 4; i++) {
		led.ledON();
		usleep(100000);
		led.ledOFF();
		usleep(100000);
	}
	*/
	//init servo
	if (!servo_init(X_PIN, RANGE)) {
		cout << "yaw servo init failed" << endl;
		return -1;
	}
	if (!servo_init(Y_PIN, RANGE)) {
		cout << "pitch servo init failed" << endl;
		return -1;
	}
	PIDctrl pidX(aPx, aIx, aDx, 20);
	PIDctrl pidY(aPy, aIy, aDy, 20);//init PID
	double error_x = 0.0, error_y = 0.0;

	PointKF KF;//init Kalman Filter
	KF.kalmanInit();

	//init tracker//////////////////
	String tracker_algorithm = "KCF";
	//String video_name = parser.get<String>( 1 );
	Ptr<Tracker> tracker_left;
	Ptr<Tracker> tracker_right;

	//init camera,0-left/1-right
	VideoCapture cam_left(0);
	VideoCapture cam_right(1);

	cout << "find camera ok" << endl;

	cam_left.set(CV_CAP_PROP_FRAME_WIDTH, 320);
	cam_left.set(CV_CAP_PROP_FRAME_HEIGHT, 240);

	cam_right.set(CV_CAP_PROP_FRAME_WIDTH, 320);
	cam_right.set(CV_CAP_PROP_FRAME_HEIGHT, 240);

	//cam_left.set(CV_CAP_PROP_MODE, 1);
	//cam_right.set(CV_CAP_PROP_MODE, 1);

	cam_left.set(CV_CAP_PROP_FOURCC, CV_FOURCC('M', 'J', 'P', 'G'));
	cam_right.set(CV_CAP_PROP_FOURCC, CV_FOURCC('M', 'J', 'P', 'G'));

	cout << "camera set ok" << endl;

	//如果打开失败，报错
	if (!cam_left.isOpened() || !cam_right.isOpened()) {
		cout << "camera open error" << endl;
		return -1;

	}
	cout << "camera open ok" << endl;
	//init camera params
	Config stereo(argv[1]);
	cout << "stereo config ok" << endl;
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

	//初始化级联分类器，这里以opencv自带的人脸识别的xml文件为例
	CascadeClassifier cascade;
	cascade.load(argv[2]);
	cout << "cascade load ok" << endl;
	//init values
	Mat pic_left, pic_right;
	Mat pic_left_rect, pic_right_rect;
	//Mat gray_left, gray_right;
	vector<Rect> target_left;
	vector<Rect> target_right;
	//Rect2d target_left_box;
	//Rect2d target_right_box;
	bool right_detected = false;	//detected flag;
	bool left_detected = false;
	bool tracker_initialized = false;	//
	//vector<KeyPoint> keypoints_left, keypoints_right;
	//Mat desciptors_left, descriptors_right;

	//Ptr<ORB> orb = ORB::create(500);

	//vector<DMatch> matches;
	//BFMatcher matcher(NORM_HAMMING);

	int inside_loop = 0;
	cout << "begin video" << endl;
	clock_t start, finish;
	start = clock();
	finish = clock();
	int outside_loop = 0;
	while (1) {
		//视频流输入图片
		cam_left >> pic_left;
		cam_right >> pic_right;
		//多少个周期一检测，视情况调整
		if (outside_loop % 50 == 0) {
			outside_loop = 0;
			tracker_initialized = false;
			tracker_left = Tracker::create(tracker_algorithm);
			tracker_right = Tracker::create(tracker_algorithm);
			if (tracker_left == NULL) {
				cout
						<< "***Error in the instantiation of the left tracker...***\n";
				continue;
			}
			if (tracker_right == NULL) {
				cout
						<< "***Error in the instantiation of the right tracker...***\n";
				continue;
			}
		}
		outside_loop++;
		if (inside_loop % 10 == 0) {
			cout << "time: " << (float) (finish - start) / CLOCKS_PER_SEC
					<< " (s) " << endl;
			start = clock();
			inside_loop = 0;
			//先对图片做矫正
			stereo.doRectifyL(pic_left, pic_left_rect);
			stereo.doRectifyR(pic_right, pic_right_rect);

			if (!tracker_initialized) {
				//检测是否有目标
				//灰度化，均衡化
				//cvtColor(pic_left_rect, gray_left, COLOR_BGR2GRAY);
				//equalizeHist(gray_left, gray_left);
				//cvtColor(pic_right_rect, gray_right, COLOR_BGR2GRAY);
				//equalizeHist(gray_right, gray_right);

				cascade.detectMultiScale(pic_left_rect, target_left, 1.1, 2,
						0 | CASCADE_SCALE_IMAGE, Size(30, 30));
				cascade.detectMultiScale(pic_right_rect, target_right, 1.1, 2,
						0 | CASCADE_SCALE_IMAGE, Size(30, 30));
			}
			if (target_left.size() == 1) {	//if left no target, continue
				left_detected = true;
			} else {
				left_detected = false;
				finish = clock();
				continue;
			}
			if (target_right.size() == 1) {
				right_detected = true;
			} else {
				right_detected = false;
			}
			if (!tracker_initialized && left_detected && right_detected) {//initializes the tracker//only do one time
				cout << "can init tracker" << endl;
				/*
				 Rect2d target_left_box((double) target_left[0].x,
				 (double) target_left[0].y,
				 (double) target_left[0].width,
				 (double) target_left[0].height);
				 Rect2d target_right_box((double) target_right[0].x,
				 (double) target_right[0].y,
				 (double) target_right[0].width,
				 (double) target_right[0].height);
				 */
				Rect2d target_left_box1((double) target_left[0].x,
						(double) target_left[0].y,
						(double) target_left[0].width,
						(double) target_left[0].height);
				Rect2d target_right_box1((double) target_right[0].x,
						(double) target_right[0].y,
						(double) target_right[0].width,
						(double) target_right[0].height);
				if (!tracker_left->init(pic_left_rect, target_left_box1)) {
					cout << "***Could not initialize left tracker...***\n";
					left_detected = false;
					finish = clock();
					continue;
				}
				if (!tracker_right->init(pic_right_rect, target_right_box1)) {
					cout << "***Could not initialize right tracker...***\n";
					right_detected = false;
					finish = clock();
					continue;
				}
				tracker_initialized = true;
				finish = clock();
				cout << "initialize tracker ok" << endl;
				continue;
			} else if (tracker_initialized) {
				//updates the tracker

				Rect2d target_left_box;
				Rect2d target_right_box;

				Point2d center_left;
				Point2d center_right;
				cout << "define center ok" << endl;
				if (tracker_left->update(pic_left_rect, target_left_box)) {
					rectangle(pic_left_rect, target_left_box, Scalar(255, 0, 0),
							2, 1);
					cout << "begin center ok" << endl;
					center_left.x = target_left_box.x
							+ cvRound(target_left_box.width / 2.0);
					center_left.y = target_left_box.y
							+ cvRound(target_left_box.height / 2.0);

					cout << "set left center ok" << endl;
					left_detected = true;
				} else {
					left_detected = false;
				}
				if (tracker_right->update(pic_right_rect, target_right_box)) {
					rectangle(pic_right_rect, target_right_box,
							Scalar(255, 0, 0), 2, 1);

					center_right.x = target_right_box.x
							+ cvRound(target_right_box.width / 2.0);
					center_right.y = target_right_box.y
							+ cvRound(target_right_box.height / 2.0);

					cout << "set right center ok" << endl;
					right_detected = true;
				} else {
					right_detected = false;
				}

				//如果两边各有一个，认为是同一个
				if (left_detected && right_detected) {
					led.ledON();	//两边都检测到，亮灯
					cout << "one target, and detected by two camera" << endl;
					//提取target所在区域的特征点
					//orb->detect(pic_left_rect(target_left_box), keypoints_left);
					//orb->detect(pic_right_rect(target_right_box),
					//		keypoints_right);

					//将特征点坐标值加上目标框的坐标值，回归到整个图像的坐标
					//for (size_t i = 0; i < keypoints_left.size(); i++) {
					//	keypoints_left[i].pt = keypoints_left[i].pt
					//			+ Point2f(target_left_box.x, target_left_box.y);
					//}
					//for (size_t i = 0; i < keypoints_right.size(); i++) {
					//	keypoints_right[i].pt = keypoints_right[i].pt
					//			+ Point2f(target_right_box.x,
					//					target_right_box.y);
					//}
					//在图像中计算描述子
					//orb->compute(pic_left_rect, keypoints_left,
					//		desciptors_left);
					//orb->compute(pic_right_rect, keypoints_right,
					//		descriptors_right);

					//对特征点进行匹配
					//matcher.match(desciptors_left, descriptors_right, matches);

					//double max_dist = 0;
					//for (size_t i = 0; i < desciptors_left.rows; i++) {
					//	double dist = matches[i].distance;
					//	if (dist > max_dist)
					//		max_dist = dist;
					//}
					//选取好的匹配
					//vector<DMatch> good_matches;
					//for (size_t i = 0; i < desciptors_left.rows; i++) {
					//	if (matches[i].distance <= 0.5 * max_dist)
					//		good_matches.push_back(matches[i]);
					//}

					////好的匹配点少，结束
					//if (good_matches.size() < 5) {
					//	cout << "too small matches, use only left camera"
					//			<< endl;

					//	vector<Point2d> pts_left;
					//	for (auto k : keypoints_left) {

					//对好的匹配，把像素坐标映射到相机坐标系
					//vector<Point2d> pts_1, pts_2;
					//for (DMatch m : good_matches) {
					//	pts_1.push_back(
					//			pixel2cam(keypoints_left[m.queryIdx].pt,
					//stereo.K_l
					//));
					//	pts_2.push_back(
					//			pixel2cam(keypoints_right[m.trainIdx].pt,
					//					stereo.K_r));
					//}
					//opencv里自带的算3d坐标的函数，计算匹配点对在左相机坐标系下的3d坐标（不是归一化的，是齐次坐标系）

					Mat pts_4d;
					vector<Point2d> c_right;
					vector<Point2d> c_left;
					c_left.push_back(center_left);
					c_right.push_back(center_right);
					triangulatePoints(T1, T2, c_left, c_right, pts_4d);
					//齐次的转化为非其次的
					vector<Point3d> points;
					Mat x = pts_4d.col(0);
					x /= x.at<double>(3, 0);
					Point3d p(x.at<double>(0, 0), x.at<double>(1, 0),
							x.at<double>(2, 0));
					points.push_back(p);

					//对计算的物体特征点求均值，视为物体3d坐标
					double x_left = 0, y_left = 0, z_left = 0;
					x_left = points[0].x;
					y_left = points[0].y;
					z_left = points[0].z;
					//for (int i = 0; i < points.size(); i++) {
					//	x_left += points[i].x;
					//	y_left += points[i].y;
					//	z_left += points[i].z;
					//}
					//x_left /= points.size();
					//y_left /= points.size();
					//z_left /= points.size();
					cout << "target 3d corrdinate:" << endl << x_left << "	"
							<< y_left << "	" << z_left << endl;
					cout << endl;
					//start control
					error_x= x_left/z_left*180/3.1416;
					error_y= y_left/z_left*180/3.1416;
					// pid input is angle error
					KF.kalmanPredict(error_x, error_y);
					error_x = KF.predict_pt.x;
					error_y = KF.predict_pt.y;
					pidX.calc(error_x);
					pidY.calc(error_y);
					servo_turn(X_PIN,pidX.output);
					servo_turn(Y_PIN,pidY.output);
				}
				//如果只有左边检测到，给个可能的区域，物体在那个射线上
				else if (left_detected && !right_detected) {	//only left
					cout
							<< "left detect and right not detect, use only left camera"
							<< endl;

					vector<Point2d> c_left;
					c_left.push_back(pixel2cam(center_left, stereo.K_l));
					double x_left = 0, y_left = 0, z_left = 1000;
					x_left = c_left[0].x;
					y_left = c_left[0].y;

					cout << "target 3d corrdinate:" << endl << x_left * 1000
							<< "		" << y_left * 1000 << "		" << z_left << endl;

				} else if (!left_detected && !right_detected) {	//no left and no right
					cout << "no target，wait for next" << endl;
				}

			}
			cout << "can show" << endl;
			imshow("left", pic_left_rect);
			imshow("right", pic_right_rect);
			finish = clock();
		} else {
			inside_loop++;
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

	int pin;
	pin = atoi(argv[1]);  //第一个参数为所要控制的引脚wiringpi编号
	int i;
	float degree;
	if (!(pin >= 0 && pin <= 8)) {     //第二个参数为需要控制舵机转动的角度
		printf("only setup pin 1 to 8\n");
		exit(0);
	}
	if (!(atoi(argv[2]) >= -135 && atoi(argv[2]) <= 135)) {
		printf("degree is between -135 and 135\n");
		exit(0);
	}

	degree = 15 + atof(argv[2]) / 270.0 * 20.0;
	wiringPiSetup();  //wiringpi初始化
	softPwmCreate(pin, 15, RANGE);  //创建一个使舵机转到中心的pwm输出信号
	delay(1000);
	for (i = 0; i < 5; i++) {
		softPwmWrite(pin, 15);   //使舵机转到中心
		delayMicroseconds(1000000);
		printf("%f\n", degree);
		softPwmWrite(pin, degree);   //转到预期角度
		delayMicroseconds(1000000);
	}

	return 0;
}

bool servo_init(int pin, int range) {
	if (!(pin >= 0 && pin <= 16)) {     //第二个参数为需要控制舵机转动的角度
		cout << "only setup pin 1 to 16\n" << endl;
		return false;
	}
	wiringPiSetup();  //wiringpi初始化
	softPwmCreate(pin, 15, RANGE);  //创建一个使舵机转到中心的pwm输出信号
	return true;

}

bool servo_turn(int pin, double angle) {
	if (angle > 45 || angle < -45) {
		cout << "too large angle " << endl;
		return false;
	}
	double degree = 15 + angle / 270.0 * 20.0;
	cout << degree << endl;
	softPwmWrite(pin, degree);  //转到预期角度
	return true;
}
Point2f pixel2cam(const Point2d& p, const Mat& K) {
	return Point2f((p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
			(p.y - K.at<double>(1, 2)) / K.at<double>(1, 1));
}

Point2d cam2pixel(const Point2d& p, const Mat& K_right) {
	return Point2d(p.x * K_right.at<double>(0, 0) + K_right.at<double>(0, 2),
			p.y * K_right.at<double>(1, 1) + K_right.at<double>(1, 2));
}
