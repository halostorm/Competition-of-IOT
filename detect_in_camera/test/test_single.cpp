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
	PIDctrl pidY(aPy, aIy, aDy, 20); //init PID
	double error_x = 0.0, error_y = 0.0;

	PointKF KF; //init Kalman Filter
	KF.kalmanInit();

	//init tracker//////////////////
	String tracker_algorithm = "KCF";
	//String video_name = parser.get<String>( 1 );
	Ptr<Tracker> tracker_left;

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
	Mat pic_left;
	Mat pic_left_rect;
	//Mat gray_left, gray_right;
	vector<Rect> target_left;
	//Rect2d target_left_box;
	//Rect2d target_right_box;
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
		//多少个周期一检测，视情况调整
		if (outside_loop % 50 == 0) {
			outside_loop = 0;
			tracker_initialized = false;
			tracker_left = Tracker::create(tracker_algorithm);
			if (tracker_left == NULL) {
				cout
						<< "***Error in the instantiation of the left tracker...***\n";
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

			if (!tracker_initialized) {
				//检测是否有目标
				//灰度化，均衡化
				//cvtColor(pic_left_rect, gray_left, COLOR_BGR2GRAY);
				//equalizeHist(gray_left, gray_left);
				//cvtColor(pic_right_rect, gray_right, COLOR_BGR2GRAY);
				//equalizeHist(gray_right, gray_right);

				cascade.detectMultiScale(pic_left_rect, target_left, 1.1, 2,
						0 | CASCADE_SCALE_IMAGE, Size(30, 30));
			}
			if (target_left.size() == 1) {	//if left no target, continue
				left_detected = true;
			} else {
				left_detected = false;
				finish = clock();
				continue;
			}
			if (!tracker_initialized && left_detected) {//initializes the tracker//only do one time
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
				if (!tracker_left->init(pic_left_rect, target_left_box1)) {
					cout << "***Could not initialize left tracker...***\n";
					left_detected = false;
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
				Point2d center_left;

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

				//如果两边各有一个，认为是同一个
				if (left_detected) {
					//led.ledON();	//两边都检测到，亮灯
					cout << "detected" << endl;

					vector<Point2d> c_left;
					c_left.push_back(pixel2cam(center_left, stereo.K_l));
					double x_left = 0, y_left = 0, z_left = distance;
					x_left = c_left[0].x;
					y_left = c_left[0].y;

					cout << "target 3d corrdinate:" << endl << x_left * 1000
							<< "		" << y_left * 1000 << "		" << z_left << endl;

					//start control
					error_x = (x_left-offset_x) / z_left * 180 / 3.1416;
					error_y = (y_left-offset_y) / z_left * 180 / 3.1416;
					// pid input is angle error
					KF.kalmanPredict(error_x, error_y);
					error_x = KF.predict_pt.x;
					error_y = KF.predict_pt.y;
					pidX.calc(error_x);
					pidY.calc(error_y);
					servo_turn(X_PIN, pidX.output);
					servo_turn(Y_PIN, pidY.output);
					//如果只有左边检测到，给个可能的区域，物体在那个射线上
				} else if (!left_detected) {	//no left and no right
					cout << "no target，wait for next" << endl;
				}

			}
			cout << "can show" << endl;
			imshow("left", pic_left_rect);
			finish = clock();
		} else {
			inside_loop++;
		}
		char c = waitKey(10);
		if (c == 32) {
			imwrite("../img_left.jpg", pic_left);
		}
		if (c == 27) {
			break;
		}
		//imshow("left",pic_left);
		//imshow("right",pic_right);
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
