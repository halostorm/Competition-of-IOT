#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <unistd.h>
#include <algorithm> 
#include <opencv2/opencv.hpp>
#include <sys/time.h>


using namespace std;
using namespace cv;

#define H 480
#define W 640

#define Gray_th 220
#define Rgb_th 110

bool compare_angle(const RotatedRect rect1,const RotatedRect rect2)
{
  return (rect1.angle<rect2.angle);
}


class BaseArmorDetection
{
public:
  Mat img_gray,img_rgb;
  vector<Mat> img_bgr;  
  Rect roi;
  Point center,precenter;
  bool RedorBlue;
  
  struct timeval tv;
  vector<Point2f> pt;
  int nCount;
  
  Mat preimg;
  
  BaseArmorDetection(const Size &size,bool Red)
  {
	img_gray.create(size,CV_8UC1);
	roi=Rect(0,0,size.width,size.height);
	RedorBlue=Red;
	nCount=0;
	precenter=Point(0,0);
  }
  
  void setImg(Mat &gray,Mat &rgb);
  bool colorDetection();
  bool opticalflowDection();
  Point detect();
  void adjectRoi();
  float ptDistance(Point2f&,Point2f&);
};


void BaseArmorDetection::setImg(Mat &gray,Mat &rgb)
{
  preimg=img_rgb.clone();
  img_rgb=rgb.clone();
  
  img_gray=gray.clone();
  split(rgb,img_bgr);
  
}

bool BaseArmorDetection::colorDetection()
{
  Mat roi_gray,roi_rb;
  Mat add,rb;
  pt.clear();
  
  if (RedorBlue)
	rb=img_bgr[2]-img_bgr[0];
  else
	rb=img_bgr[0]-img_bgr[2];
  
  threshold(img_gray(roi), roi_gray, Gray_th, 255, THRESH_BINARY);	
  threshold(rb(roi), roi_rb, Rgb_th, 255, THRESH_BINARY);
  dilate(roi_gray,roi_gray,getStructuringElement( 0,Size(2,2), Point(1, 1)),Point(-1, -1),1);
  dilate(roi_rb,roi_rb,getStructuringElement( 0,Size(5,5), Point(1, 1)),Point(-1, -1),2);
  //imshow("1",roi_gray);
  //imshow("2",roi_rb);
  bitwise_and(roi_rb,roi_gray,add);//灯柱
    //  imshow("3",add);
 
  vector<vector<Point> > contours;  
  vector<vector<Point> > contours_ok; 
  vector<Vec4i> hierarchy;  
  findContours(roi_gray, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

  vector<RotatedRect> small_rect;		//Find fit courtous
  for(vector<vector<Point> >::iterator iter=contours.begin();iter!=contours.end();)
  {
	cout<<iter->size()<<"   ";
	if (iter->size()<20 || iter->size()>60)
	{
	  iter=contours.erase(iter);
	}
	else
	{
	  RotatedRect fitrect=fitEllipse(*iter);
	  cout<<"/fffff "<<fitrect.size.height<<"   "<<fitrect.size.width<<"   "<<fitrect.angle<<endl;
	  if((fitrect.size.width/fitrect.size.height >2 ||fitrect.size.height/fitrect.size.width>2)&& min(fitrect.size.width,fitrect.size.height)>2)
	  {
		small_rect.push_back(fitrect);
		
	  }
	  ++iter;
	}
  }

  waitKey(1);
  if (small_rect.size()>=4) //筛选
  {
	sort(small_rect.begin(),small_rect.end(),compare_angle);
	vector<float> angle;
	
	for(vector<RotatedRect>::iterator iter=small_rect.begin();iter!=(small_rect.end())-3;++iter)
	{
	  float a=(iter+3)->angle - iter->angle;
	  angle.push_back(a);
	}
	int min=*min_element(angle.begin(), angle.end());
	int n=distance(angle.begin(),min_element(angle.begin(), angle.end()));
	if (min<10 && abs(ptDistance(small_rect[n].center,small_rect[n+1].center)-ptDistance(small_rect[n+2].center,small_rect[n+3].center))<5)
	{
	  
	  cout<<small_rect[n].center<<small_rect[n+1].center<<small_rect[n+2].center<<small_rect[n+3].center<<endl;
	  cout<<ptDistance(small_rect[n].center,small_rect[n+1].center)<<" "<<ptDistance(small_rect[n+2].center,small_rect[n+3].center)<<endl;
	  center.x=(small_rect[n].center.x+small_rect[n+1].center.x+small_rect[n+2].center.x+small_rect[n+3].center.x)/4+roi.x;
	  center.y=(small_rect[n].center.y+small_rect[n+1].center.y+small_rect[n+2].center.y+small_rect[n+3].center.y)/4+roi.y;
	  
	  return true;
	}
	else
	{
	  center=Point(0,0);
	  //  roi=Rect(0,0,W,H);
	  return false;
	}
	
  }
  else
  {
	center=Point(0,0);
	//roi=Rect(0,0,W,H);
	return false;
  }
  
  
  
}

bool BaseArmorDetection::opticalflowDection()
{
  vector<Point2f> prePt,curPt;
  vector<uchar> status;
  vector<float> err;
  prePt.push_back(Point2f(precenter.x-roi.x,precenter.y-roi.y));
  cout<<prePt.size()<<endl;
  calcOpticalFlowPyrLK(preimg(roi),img_rgb(roi),prePt,curPt,status,err);
  center.x=curPt[0].x+roi.x;
  center.y=curPt[0].y+roi.y;
  cout<<"OP:"<<(short)status[0]<<endl;
  return (short)status[0];
}
Point BaseArmorDetection::detect()
{
  //cout<<"PPPPT:"<<center<<precenter<<endl;
  
  if (precenter.x==0)
  {
	colorDetection();
	precenter=center;
	roi=Rect(0,0,W,H);
	center=Point(0,0);
  }
  else
  {
	if(colorDetection())
	{
	  precenter=center;
	  
	}
	else
	{
	  if(!opticalflowDection()||center.x<20||center.x>620||center.y<20||center.y>460)
	  {  
		
		precenter=Point(0,0);
		roi=Rect(0,0,W,H);
		center=Point(0,0);
	  }
	  else
	  {
		precenter=center;
	  }
	  
	  
	  
	}
  }
  adjectRoi();
  return center;
  
}
void BaseArmorDetection::adjectRoi()
{
  
  int rectwidth=180;
  if (center.x!=0)
  {
	roi.width=rectwidth;
	roi.height=rectwidth;
	
	if ((center.x-rectwidth/2)<0)
	  roi.x=0;
	else if ((center.x+rectwidth/2)>W)
	  roi.x=(W-rectwidth);
	else
	  roi.x=center.x-rectwidth/2;
	
	if ((center.y-rectwidth/2)<0)
	  roi.y=0;
	else if ((center.y+rectwidth/2)>H)
	  roi.y=(H-rectwidth);
	else
	  roi.y=center.y-rectwidth/2;
	
	
  }
  else
	roi=Rect(0,0,W,H);
}

float BaseArmorDetection::ptDistance(Point2f& p1,Point2f& p2)
{
  return sqrt((p1.x-p2.x)*(p1.x-p2.x)+(p1.y-p2.y)* (p1.y-p2.y));
  
}
