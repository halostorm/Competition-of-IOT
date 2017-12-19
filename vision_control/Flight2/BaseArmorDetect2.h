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


bool compare_angle(const RotatedRect rect1,const RotatedRect rect2)
{
	return (rect1.angle<rect2.angle);
}


class BaseArmorDetection
{
public:
	Mat img_gray;
	vector<Mat> img_bgr;  
	Rect roi;
	Point center;
	int d=0,dd=0,z=0,pre_z=0;
	double dt;
	int lost_count;

	Point pre_pt;
	int pre_t;
	Point tar_pt;
	double start=0,end=0;
	
	bool color_mode;//1-red 0-blue
	
	BaseArmorDetection(const Size &size,bool red)
	{
		img_gray.create(size,CV_8UC1);
		roi=Rect(0,0,size.width,size.height);
		lost_count=0;
		tar_pt=Point(0,0);
		pre_pt=Point(0,0);
		color_mode=red;
	}
	
	void setImg(Mat &gray,Mat &rgb);
	bool armorDetection();
};


void BaseArmorDetection::setImg(Mat &gray,Mat &rgb)
{
	img_gray=gray.clone();
	split(rgb,img_bgr);
	
}

bool BaseArmorDetection::armorDetection()
{

	Mat add;
	pre_pt=center;pre_z=z;
	cout<<"ddd";
	threshold(img_gray, img_gray, 220, 255, CV_THRESH_BINARY);
	cout<<"ddd";
	Mat rb;
	if (color_mode)
		rb=img_bgr[2]-img_bgr[0];
	else
		rb=img_bgr[0]-img_bgr[2];
	threshold(rb, rb, 120, 255, THRESH_BINARY);
	dilate(rb,rb,getStructuringElement( 0,Size(5,5), Point(2, 2)),Point(-1, -1),3);
	bitwise_and(rb,img_gray,add);//灯柱

	vector<vector<Point> > contours;  
	vector<vector<Point> > contours_ok; 
	vector<Vec4i> hierarchy;  
	findContours(add(roi), contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

	vector<RotatedRect> small_rect;		//Find fit courtous
	for(vector<vector<Point> >::iterator iter=contours.begin();iter!=contours.end();)
	{
		if (iter->size()<10 || iter->size()>30)
		{
			iter=contours.erase(iter);
		}
		else
		{
			RotatedRect fitrect=fitEllipse(*iter);
			if(fitrect.size.width/fitrect.size.height >2 ||fitrect.size.height/fitrect.size.width>2)
				small_rect.push_back(fitrect);
		
			++iter;
		}
	}
	
		
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
		if (min<10)
		{
		center.x=(small_rect[n].center.x+small_rect[n+1].center.x+small_rect[n+2].center.x+small_rect[n+3].center.x)/4;
		center.y=(small_rect[n].center.y+small_rect[n+1].center.y+small_rect[n+2].center.y+small_rect[n+3].center.y)/4;
		lost_count=0;
		}
		else
		{
			if ((small_rect[n+2].angle-small_rect[n].angle)>(small_rect[n+3].angle-small_rect[n+1].angle))
			{
				center.x=(small_rect[n+1].center.x+small_rect[n+2].center.x+small_rect[n+3].center.x)/3;
				center.y=(small_rect[n+1].center.y+small_rect[n+2].center.y+small_rect[n+3].center.y)/3;\
				lost_count=0;
			}
			else
			{
				center.x=(small_rect[n].center.x+small_rect[n+1].center.x+small_rect[n+2].center.x)/3;
				center.y=(small_rect[n].center.y+small_rect[n+1].center.y+small_rect[n+2].center.y)/3;
				lost_count=0;
			}
		}
		

	}
	else if (small_rect.size()==3 && small_rect[2].angle-small_rect[0].angle<10)
	{
		
		center.x=(small_rect[0].center.x+small_rect[1].center.x+small_rect[2].center.x)/3;
		center.y=(small_rect[0].center.y+small_rect[1].center.y+small_rect[2].center.y)/3;
		lost_count=0;
		center.x-=30;
	}
	else
	{
		lost_count=lost_count+1;
		if (lost_count>2)
		{
			center=Point(0,0);
			roi=Rect(0,0,640,480);
			pre_pt=Point(0,0);
			pre_t=0;
		}
		return false;
	}
	

	int rectwidth=150;
	
		roi.width=rectwidth;
		roi.height=rectwidth;
		center.x+=roi.x;
		center.y+=roi.y;
		if ((center.x-rectwidth/2)<0)
			roi.x=0;
		else if ((center.x+rectwidth/2)>640)
			roi.x=(640-rectwidth);
		else
			roi.x=center.x-rectwidth/2;
			
		if ((center.y-rectwidth/2)<0)
			roi.y=0;
		else if ((center.y+rectwidth/2)>480)
			roi.y=(480-rectwidth);
		else
			roi.y=center.y-rectwidth/2;
		
		end=getTickCount();
		dt=(end-start)/getTickFrequency();
		start=getTickCount();
		z=sqrt((center.x-320)*(center.x-320)+(center.y-280)*(center.y-280));
		
		if (pre_pt.x!=0)
		{
			tar_pt.x=center.x+(center.x-pre_pt.x)*6;
			tar_pt.y=center.y+(center.y-pre_pt.y)*6;
			dd =sqrt((center.y-pre_pt.y)*(center.y-pre_pt.y)+(center.x-pre_pt.x)*(center.x-pre_pt.x));
			d=abs((center.y-pre_pt.y)*320-(center.x-pre_pt.x)*280+center.y*(center.x-pre_pt.x)-center.x*(center.y-pre_pt.y))/sqrt((center.y-pre_pt.y)*(center.y-pre_pt.y)+(center.x-pre_pt.x)*(center.x-pre_pt.x));
	
		}
		else
			tar_pt=center;
		
		//cout<<"Used:"<<dt<<" D："<<d<<"  Z："<<z<<"  Pre:"<<pre_z<<pre_pt<<center<<"  Tar:"<<tar_pt<<endl;
				
		return true;
	
		
}


