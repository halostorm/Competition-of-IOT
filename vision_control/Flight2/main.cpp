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
void sig_callback(int signum)
{
	printf("exit read image\n");
	RUNNING_MODE = false;
	return;
}

struct ImgData
{
	Mat img;
	unsigned int framenum;
};
ImgData image;

static void *readLoop(void *data)
{
	RMVideoCapture cap("/dev/video0",3);
	cap.setVideoFormat(640, 480, 1);
	cap.setExposureTime(0, 70);//settings->exposure_time);
	cap.startStream();
	cap.info();
	ImgData img;
	image.img.create(480, 640, CV_8UC3);
	cap >> image.img;

	double start, end;
	RUNNING_MODE = true;
	while (RUNNING_MODE)
	{
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

static void *detectLoop(void *data)
{
	//飞机初始化
	LinuxSerialDevice* serialDevice = new LinuxSerialDevice(UserConfig::deviceName, UserConfig::baudRate);
	CoreAPI* api = new CoreAPI(serialDevice);
	Flight* flight = new Flight(api);
	LinuxThread read(api, 2);
	Camera* camera = new Camera(api);
	VirtualRC* vrc = new VirtualRC(api);
	BroadcastData bcd;
	unsigned short broadcastAck = api->setBroadcastFreqDefaults(1);
	int setupStatus = setup(serialDevice, api, &read);
	int offsetx=15;
	int offsety=-40;
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

	//PID
	double dx = 0, dy = 0;
	PIDctrl pidX(aPx, aIx, aDx, 0.5);
	PIDctrl pidY(aPy, aIy, aDy, 0.5);
	PIDctrl pidX2(bPx, bIx, bDx, 0.5);
	PIDctrl pidY2(bPy, bIy, bDy, 0.5);

	
	/*pthread_t armor_thread;
	pthread_create(&armor_thread, NULL, armorLoop, (void*)fd);*/

	PointKF KF;
	KF.kalmanInit();

	double start = 0, start2 = 0, end = 0;
	static unsigned char motor_flag=1;
	api->setControl(0, 1);
	int centerCount = 0, landCount = 0, lostcount = 0;
	float tagPerimeter_tar = 270;

	//串口
	int motor;
	int last_motor;

	//int gear=-4545;
	
	while (RUNNING_MODE)
	{
		//waitKey(3000);
		end = getTickCount();
		//cout<<(end-start)/getTickFrequency()<<endl;
		start = getTickCount();
		bcd = api->getBroadcastData();
		if (bcd.rc.gear == -4545)
		{
			motor=32;
		}else{
			motor=31;
		}
		if(motor!=last_motor)
		{
			motor_flag=(char)motor;
			last_motor=motor;
			//cout<<motor_flag<<endl;
			uartSend(fd, &motor_flag, 1);
		}
		if (1/*(bcd.rc.gear == -4545*/)//Circle Detection
		{
			BASEDETECT_MODE = 0;

			pthread_rwlock_rdlock(&img_spinlock);
			image_get.img = image.img.clone();
			image_get.framenum = image.framenum;
			pthread_rwlock_unlock(&img_spinlock);
			cvtColor(image_get.img, img_gray, CV_RGB2GRAY);
			tria.preproc(img_gray);
			circledect.setImg(img_gray);
			if (circledect.circleDetection() == true)
			{
				led.ledON();
				dy = -(circledect.center.x - cam_x)+offsetx;//60
				dx = -(cam_y - circledect.center.y)+offsety;// + circledect.radius / 1.5;// 25
				r = circledect.radius;
				KF.kalmanPredict(dx,dy);
				dx=KF.predict_pt.x;
				dy=KF.predict_pt.y;
				centerCount++;
				lostcount = 0;
				//cout<< "Frame:" << image_get.framenum<<"adx:="<<dx<<" ,dy:="<<dy<<endl;
				if (LAND_MODE && bcd.rc.mode == 8000)
				{
					api->setControl(1, 1);
					//dx = cam_y - circledect.center.y + 80;
					cout << "Frame:" << image_get.framenum << " *Prepare land*  R:" << r << "  C:" << circledect.center << "  PID: " << pidX2.output << "  " << pidY2.output << "  dx: " << dx << "  dy:" << dy << "  V: " << bcd.v.x << " " << bcd.v.y << " " << bcd.v.z << "  H:" << bcd.pos.height << endl;
					pidX2.calc(dx);
					pidY2.calc(dy);

					if (r < 120)
						flight->setMovementControl(0x4B, -pidX2.output, -pidY2.output, -0.1, 0);

					if (abs(dx) < 25 && abs(dy) < 25)//15
					{
						landCount++;
						if (landCount > 4)//4
						{
							ackReturnData landingack = landing(api, flight, 1);
							LAND_MODE = false;
						}
						else
							flight->setMovementControl(0x4B, 0, 0, 0, 0);
					}
					else
					{
						landCount = 0;
						flight->setMovementControl(0x4B, -pidX2.output, -pidY2.output, 0, 0);
					}
				}
				else if (!LAND_MODE && bcd.status == 3 && bcd.rc.mode == 8000)
				{
					api->setControl(1, 1);

					pidX.calc(dx);
					pidY.calc(dy);

					if (abs(dx) < 30 && abs(dy) < 30 && abs(bcd.v.x) < 0.04 && abs(bcd.v.y) < 0.04)
					{
						flight->setMovementControl(0x4B, -pidX.output, -pidY.output, -0.1, 0);
						cout << "Frame:" << image_get.framenum << "*Adjust down*  R:" << r << "  PID: " << pidX.output << "  " << pidY.output << "  dx:" << dx << "  dy:" << dy << "  V:" << bcd.v.x << " " << bcd.v.y << endl;
						if (r > 120)
							LAND_MODE = true;
					}
					else
					{
						flight->setMovementControl(0x4B, -pidX.output, -pidY.output, 0, 0);
						cout << "Frame:" << image_get.framenum << "*Adjust*  R:" << r << "  PID: " << pidX.output << "  " << pidY.output << "  dx:" << dx << "  dy:" << dy << "  V:" << bcd.v.x << " " << bcd.v.y << endl;

					}
				}
#if SHOWIMG
				circledect.drawCircle(image_get.img);
#endif

			}
			else if (tria.triangleDetection(img_gray)&& tria.findpoint()==1)
			{
				led.ledON();
				dy = -(tria.center.x - cam_x)+offsetx;//60
				dx = -(cam_y - tria.center.y)+offsety;
				//r = circledect.radius;
				KF.kalmanPredict(dx,dy);
				dx=KF.predict_pt.x;
				dy=KF.predict_pt.y;
				lostcount = 0;
				if (LAND_MODE && bcd.rc.mode == 8000)
				{
					api->setControl(1, 1);
					//dx = cam_y - circledect.center.y + 80;
					cout << "  PID: " << pidX2.output << "  " << pidY2.output << "Frame:" << image_get.framenum<< " dx:="<< dx<< " ,dy:=" <<dy << endl;
					pidX2.calc(dx);
					pidY2.calc(dy);

					//if (r < 120)
						//flight->setMovementControl(0x4B, pidX2.output, pidY2.output, -0.1, 0);

					if (abs(dx) < 25 && abs(dy) < 25)//15
					{
						landCount++;
						if (landCount > 4)//4
						{
							ackReturnData landingack = landing(api, flight, 1);
							LAND_MODE = false;
						}
						else
							flight->setMovementControl(0x4B, 0, 0, 0, 0);
					}
					else
					{
						landCount = 0;
						flight->setMovementControl(0x4B, -pidX2.output, -pidY2.output, 0, 0);
					}
				}
				else if (!LAND_MODE && bcd.status == 3 && bcd.rc.mode == 8000)
				{
					api->setControl(1, 1);

					pidX.calc(dx);
					pidY.calc(dy);

					if (abs(dx) < 30 && abs(dy) < 30 && abs(bcd.v.x) < 0.04 && abs(bcd.v.y) < 0.04)
					{
						flight->setMovementControl(0x4B, -pidX.output, -pidY.output, -0.1, 0);
						cout << "Frame:" << image_get.framenum << "*Adjust down*  R:" << r << "  PID: " << pidX.output << "  " << pidY.output << "  dx:" << dx << "  dy:" << dy << "  V:" << bcd.v.x << " " << bcd.v.y << endl;
						//if (r > 120)
							//LAND_MODE = true;
					}
					else
					{
						flight->setMovementControl(0x4B, -pidX.output, -pidY.output, 0, 0);
						cout << "Frame:" << image_get.framenum <<  "  PID: " << pidX.output << "  " << pidY.output << "  dx:" << dx << "  dy:" << dy << "  V:" << bcd.v.x << " " << bcd.v.y << endl;

					}
				}
#if SHOWIMG
				tria.drawTriangle(image_get.img);
#endif
			}
			else
			{
				//led.ledOFF();
				dx = 0; dy = 0;
				//	pidX.reset();
				//	pidY.reset();
				lostcount++;
				//led.ledOFF();
				if(lostcount>15)led.ledOFF();
				pidX2.reset();
				pidY2.reset();
				flight->setMovementControl(0x4B, 0, 0, 0, 0);
				/*if (bcd.rc.mode == 8000 && bcd.status == 1)//Catch ball
				{
					if (bcd.rc.pitch > 8000)//电机正转
					{
						motor_flag = 0xCC;
						uartSend(fd, &motor_flag, 1);
					}
					else if (bcd.rc.roll < -8000)//电机停
					{
						motor_flag = 0xDD;
						uartSend(fd, &motor_flag, 1);
					}
					else if (bcd.rc.pitch < -8000)//电机反
					{
						motor_flag = 0xEE;
						uartSend(fd, &motor_flag, 1);
					}
					usleep(20000);
				}*/
			}


#if SHOWIMG
			imshow("img", image_get.img);
			waitKey(1);
#endif
		}
		/*else //Amror Detection
		{
			BASEDETECT_MODE = 1;//start armor detection 

			if (bcd.rc.mode == 8000)
			{
				api->setControl(1, 1);
				BASEDETECT_MODE = 2;		 //start throw
				//cout<<tagPerimeter<<"  pre: "<<tagPerimeter_pre<<endl;
				if (bcd.rc.throttle == 0 && tagPerimeter_tar - tagPerimeter > 5 && abs(tagPerimeter_pre - tagPerimeter) < 3)
				{
					flight->setMovementControl(0x4B, 0, 0, -0.05, 0);
				}
				else if (bcd.rc.throttle == 0 && tagPerimeter - tagPerimeter_tar > 10 && abs(tagPerimeter_pre - tagPerimeter) < 3)
				{
					flight->setMovementControl(0x4B, 0, 0, 0.05, 0);
				}
				else if (bcd.rc.yaw < -3000)
				{
					motor_flag = 0xAA;
					uartSend(fd, &motor_flag, 1);
					usleep(1200000);
					motor_flag = 0xBB;
					uartSend(fd, &motor_flag, 1);
				}
				else
				{
					flight->setMovementControl(0x4B, 0, 0, 0, 0);
					if (tagPerimeter != 0)
						tagPerimeter_tar = tagPerimeter;
				}
			}
			else
			{
				api->setControl(0, 1);

			}

			pidX.reset();
			pidY.reset();
			dx = 0; dy = 0;

		}*/

	}



	int cleanupStatus = cleanup(serialDevice, api, flight, &read);
	if (cleanupStatus == -1)
	{
		cout << "Unable to cleanly destroy OSDK infrastructure. There may be residual objects in the system memory.\n";
		return 0;
	}
	cout << "Program exited successfully." << endl;
}



static void *tagsLoop(void *data)
{
	Mat img_tags;
	aruco::MarkerDetector MDetector;
	vector< aruco::Marker > markers;
	aruco::Dictionary::DICT_TYPES  dict = aruco::Dictionary::DICT_TYPES::TAG16h5;
	Mat Homo;
	float perimeter; Mat mapp(500, 500, CV_8UC1);

	MDetector.setDictionary(dict);//sets the dictionary to be employed (ARUCO,APRILTAGS,ARTOOLKIT,etc)
	MDetector.setThresholdParams(10, 9);
	MDetector.setThresholdParamRange(2, 0);
	vector<cv::Point2f> Points2D;
	vector<cv::Point2f> Points2Ddst;
	vector<cv::Point2f> Points2Ddd;

	while (!RUNNING_MODE)
		sleep(2);

	while (RUNNING_MODE)
	{
		/*pthread_rwlock_rdlock(&img_spinlock);
		img_tags = image.img.clone();
		pthread_rwlock_unlock(&img_spinlock);
		Points2D.clear();
		Points2Ddst.clear();
		perimeter = 0;
		if (BASEDETECT_MODE > 0) {
			markers = MDetector.detect(img_tags);
			cout << "Marker Find:" << markers.size() << endl;

			if (markers.size() > 1)
			{
				for (unsigned int i = 0; i < markers.size(); i++)
				{
					cout << markers[i].id << "detect" << "   Perimeter" << markers[i].getPerimeter() << endl;
					perimeter += markers[i].getPerimeter();
					markers[i].draw(img_tags, Scalar(0, 0, 255));
					if (markers[i].id == 0)
					{
						Points2D.push_back(markers[i][2]);
						Points2Ddst.push_back(Point2f(264, 1723));

						Points2D.push_back(markers[i][0]);
						Points2Ddst.push_back(Point2f(40, 1945));
					}
					else if (markers[i].id == 1)
					{
						Points2D.push_back(markers[i][2]);
						Points2Ddst.push_back(Point2f(264, 880));

						Points2D.push_back(markers[i][0]);
						Points2Ddst.push_back(Point2f(40, 1104));
					}
					else if (markers[i].id == 2)
					{
						Points2D.push_back(markers[i][2]);
						Points2Ddst.push_back(Point2f(264, 40));

						Points2D.push_back(markers[i][0]);
						Points2Ddst.push_back(Point2f(40, 262));
					}
					else  if (markers[i].id == 3)
					{
						Points2D.push_back(markers[i][2]);
						Points2Ddst.push_back(Point2f(1112, 1723));

						Points2D.push_back(markers[i][0]);
						Points2Ddst.push_back(Point2f(887, 1945));
					}
					else if (markers[i].id == 4)
					{
						Points2D.push_back(markers[i][2]);
						Points2Ddst.push_back(Point2f(1112, 40));

						Points2D.push_back(markers[i][0]);
						Points2Ddst.push_back(Point2f(887, 262));
					}
					else if (markers[i].id == 5)
					{
						Points2D.push_back(markers[i][2]);
						Points2Ddst.push_back(Point2f(1960, 1723));

						Points2D.push_back(markers[i][0]);
						Points2Ddst.push_back(Point2f(1737, 1945));
					}
					// 		  else if (markers[i].id==6)
					// 		  {  
					// 			Points2D.push_back(markers[i][2]);
					// 			Points2Ddst.push_back(Point2f(1960,880));
					// 			
					// 			Points2D.push_back(markers[i][0]);
					// 			Points2Ddst.push_back(Point2f(1737,1104));
					// 		  }
					// 		  else if (markers[i].id==7)
					// 		  {  
					// 			Points2D.push_back(markers[i][2]);
					// 			Points2Ddst.push_back(Point2f(1960,40));
					// 			
					// 			Points2D.push_back(markers[i][0]);
					// 			Points2Ddst.push_back(Point2f(1737,262));
					// 		  }

				}
				tagPerimeter_pre = tagPerimeter;
				tagPerimeter = perimeter / markers.size();

				if (Points2D.size() > 3)
				{
					Homo = findHomography(Points2D, Points2Ddst, 0);
					pthread_spin_lock(&tags_spinlock);
					Homo.copyTo(Homography);
					pthread_spin_unlock(&tags_spinlock);


				}
#if SHOWIMG
				if (Points2D.size() > 3)
				{
					perspectiveTransform(Points2D, Points2Ddd, Homo);
				}

				for (int i = 0; i < Points2Ddd.size(); i++)
					if (Points2Ddd[i].x < 2000 && Points2Ddd[i].x>0 && Points2Ddd[i].y < 2000 && Points2Ddd[i].y>0)
					{
						circle(mapp, Point(Points2Ddd[i].x / 4, Points2Ddd[i].y / 4), 3, Scalar(255), 2);
					}
				imshow("tags", img_tags);
				waitKey(1);
#endif
			}
		}
		else {
			sleep(1);
		}*/
	}

	cout << "tags exit" << endl;
}

int main(int argc, char **argv)
{
	streambuf* coutBuf = cout.rdbuf();
#if USE_LOGFILE	
	ofstream file("out.log");
	streambuf* buf = file.rdbuf();
	cout.rdbuf(buf);
#endif

	signal(SIGINT, sig_callback);

	cout << "For x: P-" << aPx << "  I-" << aIx << "  D-" << aDx << "\nFor y: P-" << aPy << "  I-" << aIy << "  D-" << aDy << endl;

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

	if (0 != geteuid())
	{
		printf("Please run ./test as root!\n");

		return -1;
	}


	if (pthread_create(&read_thread, NULL, readLoop, NULL) != 0)
	{
		printf("Read_thread create");
		return -1;
	}

	/*if (pthread_create(&tags_thread, NULL, tagsLoop, NULL) != 0)
	{
		printf("tag_thread create");
		return -1;
	}*/


	if (pthread_create(&detect_thread, NULL, detectLoop, NULL) != 0)
	{
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
