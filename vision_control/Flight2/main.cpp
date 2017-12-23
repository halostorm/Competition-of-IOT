//System Header
#include <iostream>
#include <string>
#include <cstring>
#include <pthread.h>
#include <unistd.h>
#include <fstream>
#include <semaphore.h>
#include <stdio.h>
#include <signal.h>
#include <opencv2/opencv.hpp>

//DJI Linux Application Headers
#include "LinuxSerialDevice.h"
#include "LinuxThread.h"
#include "LinuxSetup.h"
#include "LinuxCleanup.h"
#include "ReadUserConfig.h"
#include "LinuxFlight.h"
#include "LinuxWaypoint.h"
#include "LinuxCamera.h"

//DJI OSDK Library Headers
#include "DJI_API.h"

//User Library Headers
#include "CircleDetect.h"
#include "TriangleDetect.hpp"
#include "uart.h"
#include "PID.h"
#include "BaseArmorDetect.h"
#include "LedController.h"
#include "para.h"
#include "/home/ubuntu/Downloads/aruco-aruco-git/src/aruco.h"
#include "RMVideoCapture.hpp"
#include "Predictor.hpp"
#include "KFT.hpp"

using namespace std;
using namespace DJI;
using namespace DJI::onboardSDK;

//DEBUG parameter
#define SHOWIMG 0
#define USE_LOGFILE 1

pthread_rwlock_t img_spinlock;
pthread_spinlock_t tags_spinlock;
Mat Homography;
float tagPerimeter, tagPerimeter_pre;

int r = 0;

bool RUNNING_MODE = false;
bool SHUT_MODE = false;
bool LAND_MODE = false;
int BASEDETECT_MODE = 0;
char device[] = "/dev/ttyS0";
int fd;
void sig_callback(int signum) {
	printf("exit read image\n");
	RUNNING_MODE = false;
	return;
}

struct ImgData {
	Mat img;
	unsigned int framenum;
};
ImgData image;

static void *readLoop(void *data) {
	RMVideoCapture cap("/dev/video0", 3);
	cap.setVideoFormat(640, 480, 1);
	cap.setExposureTime(0, 70); //settings->exposure_time);
	cap.startStream();
	cap.info();
	ImgData img;
	image.img.create(480, 640, CV_8UC3);
	cap >> image.img;

	double start, end;
	RUNNING_MODE = true;
	while (RUNNING_MODE) {
		cap >> img.img;
		img.framenum = cap.getFrameCount();
		pthread_rwlock_wrlock(&img_spinlock);
		image.img = img.img.clone();
		image.framenum = img.framenum;
		pthread_rwlock_unlock(&img_spinlock);
	}
	printf("get_images_loop thread exit! \n");
	cap.closeStream();
}

/*static void *armorLoop(void *data)
 {
 int fd = (int)data;
 cout << fd;

 static unsigned char motor_flag;
 while (RUNNING_MODE)
 {
 if (bcd.rc.gear == -4545)
 {
 motor_flag = 0xAA;
 uartSend(fd, &motor_flag, 1);
 }
 else
 {
 motor_flag = 0xBB;
 uartSend(fd, &motor_flag, 1);
 }
 }
 }
 */

static void *detectLoop(void *data) {
	int offsetx = 15;
	int offsety = -40;
	//图像初始化
	ImgData image_get;
	Mat img_gray;
	CircleDetect circledect(Size(640, 480), 1);
	TriangleDetect tria;
	LedController led;
	for (int i = 0; i < 4; i++) {
		led.ledON();
		usleep(100000);
		led.ledOFF();
		usleep(100000);
	}
	led.ledOFF();

	//PID init
	double dx = 0, dy = 0;
	PIDctrl pidX(aPx, aIx, aDx, 0.5);
	PIDctrl pidY(aPy, aIy, aDy, 0.5);
	PIDctrl pidX2(bPx, bIx, bDx, 0.5);
	PIDctrl pidY2(bPy, bIy, bDy, 0.5);

	/*pthread_t armor_thread;
	 pthread_create(&armor_thread, NULL, armorLoop, (void*)fd);*/

	PointKF KF;
	KF.kalmanInit();

	//int gear=-4545;

	while (1) {
		//waitKey(3000);
		end = getTickCount();
		//cout<<(end-start)/getTickFrequency()<<endl;
		start = getTickCount();

		if (1) //Circle Detection
		{
			image_get.img = image.img.clone();
			image_get.framenum = image.framenum;
			cvtColor(image_get.img, img_gray, CV_RGB2GRAY);
			tria.preproc(img_gray);
			circledect.setImg(img_gray);
			if (circledect.circleDetection() == true) {
				led.ledON();
				dy = -(circledect.center.x - cam_x) + offsetx; //60
				dx = -(cam_y - circledect.center.y) + offsety; // + circledect.radius / 1.5;// 25
				r = circledect.radius;
				KF.kalmanPredict(dx, dy);
				dx = KF.predict_pt.x;
				dy = KF.predict_pt.y;
				centerCount++;
				lostcount = 0;
				//cout<< "Frame:" << image_get.framenum<<"adx:="<<dx<<" ,dy:="<<dy<<endl;
				pidX2.calc(dx);
				pidY2.calc(dy);

				pidX.calc(dx);
				pidY.calc(dy);

			}
#if SHOWIMG
			circledect.drawCircle(image_get.img);
#endif

		} else if (tria.triangleDetection(img_gray) && tria.findpoint() == 1) {

#if SHOWIMG
			tria.drawTriangle(image_get.img);
#endif
		} else {
			//led.ledOFF();
			dx = 0;
			dy = 0;
			//	pidX.reset();
			//	pidY.reset();
			lostcount++;
			//led.ledOFF();
			if (lostcount > 15)
				led.ledOFF();
			pidX2.reset();
			pidY2.reset();

#if SHOWIMG
			imshow("img", image_get.img);
			waitKey(1);
#endif
		}

	}
}
int main(int argc, char **argv) {
	streambuf* coutBuf = cout.rdbuf();
#if USE_LOGFILE	
	ofstream file("out.log");
	streambuf* buf = file.rdbuf();
	cout.rdbuf(buf);
#endif

	cout << "For x: P-" << aPx << "  I-" << aIx << "  D-" << aDx
			<< "\nFor y: P-" << aPy << "  I-" << aIy << "  D-" << aDy << endl;

	pthread_attr_t attr;
	struct sched_param schedparam;
	pthread_t read_thread;
	pthread_t detect_thread;
	pthread_t tags_thread;

	pthread_rwlock_init(&img_spinlock, 0);
	pthread_spin_init(&tags_spinlock, 0);
	fd = uartOpen(device);
	if (fd == -1)
		printf("device error");
	uartSet(fd);

	if (0 != geteuid()) {
		printf("Please run ./test as root!\n");

		return -1;
	}

	if (pthread_create(&read_thread, NULL, readLoop, NULL) != 0) {
		printf("Read_thread create");
		return -1;
	}

	/*if (pthread_create(&tags_thread, NULL, tagsLoop, NULL) != 0)
	 {
	 printf("tag_thread create");
	 return -1;
	 }*/

	if (pthread_create(&detect_thread, NULL, detectLoop, NULL) != 0) {
		printf("detect_thread create");
		return -1;
	}

	pthread_join(detect_thread, NULL);/*wait for read_thread exit*/
	pthread_join(read_thread, NULL);/*wait for read_thread exit*/

	sleep(3);

#if USE_LOGFILE    
	file.flush();
	file.close();
	time_t t;
	time(&t);
	rename("out.log", ctime(&t));
#endif

	cout.rdbuf(coutBuf);

	return 0;
}
