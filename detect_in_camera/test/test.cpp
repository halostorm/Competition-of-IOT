#include <iostream>
#include "my_refind/config.h"

//像素坐标系和相机坐标系（归一化的，就是说d=1）的转化
Point2f pixel2cam(const Point2d& p, const Mat& K);
Point2d cam2pixel(const Point2d& p, const Mat& K_right);


int main(int argc, char **argv) 
{
	if(argc!=3)
	{
		cerr<<"usage: ./area_reprojected  stereo_camera_config target_detect_config"<<endl;
		return -1;
	}
	
	//打开两个相机，先插左边的，再插右边的，设置分辨率
	VideoCapture cam_left(0);
	VideoCapture cam_right(1);
	
	cam_left.set(CV_CAP_PROP_FRAME_WIDTH,640);
	cam_left.set(CV_CAP_PROP_FRAME_HEIGHT,480);
	
	cam_right.set(CV_CAP_PROP_FRAME_WIDTH,640);
	cam_right.set(CV_CAP_PROP_FRAME_HEIGHT,480);
	
	//如果打开失败，报错
	if(!cam_left.isOpened()||!cam_right.isOpened())
	{
	  cout<<"camera open error"<<endl;
	  return -1;
	  
	}
	
	//读入相机参数，这个要按照你标定的来，例子给的是1080p的，肯定不行
	Config stereo(argv[3]);
	Mat T1 = (Mat_<double> (3,4) <<
        1,0,0,0,
        0,1,0,0,
        0,0,1,0);
	Mat T2 = (Mat_<double> (3,4) <<
        stereo.L2R_R.at<double>(0,0),stereo.L2R_R.at<double>(0,1),stereo.L2R_R.at<double>(0,2),stereo.L2R_T.at<double>(0,0),
        stereo.L2R_R.at<double>(1,0),stereo.L2R_R.at<double>(1,1),stereo.L2R_R.at<double>(1,2),stereo.L2R_T.at<double>(1,0),
        stereo.L2R_R.at<double>(2,0),stereo.L2R_R.at<double>(2,1),stereo.L2R_R.at<double>(2,2),stereo.L2R_T.at<double>(2,0));
	
	Mat R = (Mat_<double>(3,3)<<
		stereo.L2R_R.at<double>(0,0),stereo.L2R_R.at<double>(0,1),stereo.L2R_R.at<double>(0,2),
		stereo.L2R_R.at<double>(1,0),stereo.L2R_R.at<double>(1,1),stereo.L2R_R.at<double>(1,2),
		stereo.L2R_R.at<double>(2,0),stereo.L2R_R.at<double>(2,1),stereo.L2R_R.at<double>(2,2));
	
	Mat T = (Mat_<double>(3,1)<<
		stereo.L2R_T.at<double>(0,0),stereo.L2R_T.at<double>(1,0),stereo.L2R_T.at<double>(2,0)
	);
	
	//初始化级联分类器，这里以opencv自带的人脸识别的xml文件为例
	CascadeClassifier cascade;
	cascade.load(argv[4]);
	
	//初始化参数
	Mat pic_left,pic_right;
	Mat pic_left_rect,pic_right_rect;
	Mat gray_left,gray_right;
	
	vector<KeyPoint>keypoints_left,keypoints_right;
	Mat desciptors_left,descriptors_right;
	
	Ptr<ORB> orb=ORB::create(500);
	
	vector<DMatch> matches;
	BFMatcher matcher (NORM_HAMMING);
	
	
	int loop=0;
	
	while(1)
	{
	  //视频流输入图片
	  cam_left>>pic_left;
	  cam_right>>pic_right;
	  
	  //多少个周期一检测，视情况调整
	  if(loop%30==0)
	  {
	    loop=0;
	    
	    //先对图片做矫正
	    stereo.doRectifyL(pic_left,pic_left_rect);
	    stereo.doRectifyR(pic_right,pic_right_rect);
	    
	    //灰度化，均衡化
	    cvtColor(pic_left_rect,gray_left,COLOR_BGR2GRAY);
	    equalizeHist(gray_left,gray_left);
	    cvtColor(pic_right_rect,gray_right,COLOR_BGR2GRAY);
	    equalizeHist(gray_right,gray_right);
	    
	    //检测是否有目标
	    vector<Rect>target_left;
	    cascade.detectMultiScale(gray_left,target_left,1.1,2,0| CASCADE_SCALE_IMAGE, Size(30,30));
	    vector<Rect>target_right;
	    cascade.detectMultiScale(gray_right,target_right,1.1,2,0| CASCADE_SCALE_IMAGE, Size(30,30));
	    
	    //如果没有
	     if(target_left.size()==0||target_right.size()==0)
	     {
		cout<<"no target, wait for next detect time"<<endl;
	     }
	     
	     //如果两边各有一个，认为是同一个
	    else if(target_left.size()==1 && target_right.size()==1)
	     {
		cout<<"one target, and detected by two camera"<<endl;
		
		//提取target所在区域的特征点
		orb->detect(pic_left_rect(target_left[0]),keypoints_left);
		orb->detect(pic_right_rect(target_right[0]),keypoints_right);
		
		//将特征点坐标值加上目标框的坐标值，回归到整个图像的坐标
		for(size_t i=0; i<keypoints_left.size(); i++)
		{
			keypoints_left[i].pt=keypoints_left[i].pt + Point2f(target_left[0].x,target_left[0].y);
		}
		for(size_t i=0; i<keypoints_right.size(); i++)
		{
			keypoints_right[i].pt=keypoints_right[i].pt + Point2f(target_right[0].x,target_right[0].y);
		}
		//在图像中计算描述子
		orb->compute(pic_left,keypoints_left,desciptors_left);
		orb->compute(pic_right_rect,keypoints_right,descriptors_right);
		
		//对特征点进行匹配
		matcher.match(desciptors_left,descriptors_right,matches);
		
		double max_dist=0;
		for(size_t i=0; i<desciptors_left.rows; i++)
		{
		  double dist = matches[i].distance;
		  if(dist>max_dist)
		      max_dist=dist;
		 }
		//选取好的匹配
		vector<DMatch> good_matches;
		for(size_t i=0; i<desciptors_left.rows; i++)
		{
		  if(matches[i].distance <= 0.5*max_dist)
		  good_matches.push_back(matches[i]);
		}
	
		//好的匹配点少，结束
		if(good_matches.size()<5)
		{
		  cout<<"too small matches, wait for next detect"<<endl;
		}
		
		else
		{
		 //对好的匹配，把像素坐标映射到相机坐标系
		  vector<Point2d> pts_1,pts_2;
		  for(DMatch m:good_matches)
		  {
		    pts_1.push_back(pixel2cam(keypoints_left[m.queryIdx].pt,stereo.K_l));
		    pts_2.push_back(pixel2cam(keypoints_right[m.trainIdx].pt,stereo.K_r));
		  }
		  //opencv里自带的算3d坐标的函数，计算匹配点对在左相机坐标系下的3d坐标（不是归一化的，是齐次坐标系）
		  Mat pts_4d;
		  triangulatePoints(T1,T2,pts_1,pts_2,pts_4d);
		  
		  //齐次的转化为非其次的
		  vector<Point3d> points;
		  for(int i=0; i<pts_4d.cols;i++)
		  {
		    Mat x=pts_4d.col(i);
		    x/=x.at<double>(3,0);
		    Point3d p(
			  x.at<double>(0,0),
			  x.at<double>(1,0),
			   x.at<double>(2,0)
			  );
		      points.push_back(p);
		  }
		  
		  //对计算的物体特征点求均值，视为物体3d坐标
		  double x_left=0,y_left=0,z_left=0;
		  for(int i=0;i<points.size();i++)
		  {
		    x_left+=points[i].x;
		    y_left+=points[i].y;
		    z_left+=points[i].z;
		   }
		  x_left/=points.size();
		  y_left/=points.size();
		  z_left/=points.size();
		  cout<<"target 3d corrdinate:"<<endl<<x_left<<"		"<<y_left<<"		"<<z_left<<endl;
		  cout<<endl;
		  
		}
		
	     }

	     else if(target_left.size()==1 && target_right.size()==0)
	     {
	       cout<<"left detect and right not detect";
	       
	    }
	     else if(target_left.size()==0 && target_right.size()==1)
	     {
	       cout<<"right detect and left not detect";
	       
	    }

	    else
	    {
		cout<<"two much target";
	    }
	    
	  }  
	  else
	  {
	    loop++;
	  }
	  char c;
	  if(c==32)
	  {
	    imwrite("./img_left.jpg",pic_left);
	    imwrite("./img_right.jpg",pic_right);
	  }
	  
	}
    return 0;
}

Point2f pixel2cam ( const Point2d& p, const Mat& K )
{
    return Point2f
    (
        ( p.x - K.at<double>(0,2) ) / K.at<double>(0,0), 
        ( p.y - K.at<double>(1,2) ) / K.at<double>(1,1) 
    );
}

Point2d cam2pixel(const Point2d& p, const Mat& K_right)
{
	return Point2d
	(
		p.x*K_right.at<double>(0,0)+K_right.at<double>(0,2),
		p.y*K_right.at<double>(1,1)+K_right.at<double>(1,2)
	);
}
