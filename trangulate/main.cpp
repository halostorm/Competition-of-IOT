#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/video.hpp>
#include <opencv2/objdetect.hpp>

using namespace std;
using namespace cv;

//像素坐标到归一化的相机坐标系
Point2f pixel2cam(const Point2d& p, const Mat& K);

//双目矫正类，读入双目相机参数文件，对图像进行矫正
class Rectify
{
public:
	Rectify(const string& config);
	void doRectifyL(const Mat& img_src, Mat& img_rect);
	void doRectifyR(const Mat& img_src, Mat& img_rect);
	
	Mat K_l,K_r,P_l,P_r,R_l,R_r,D_l,D_r;
	int rows_l,cols_l,rows_r,cols_r;
	
	
	Mat M1l,M1r,M2l,M2r;
};

//读入分类配置文件，在图像中搜索感兴趣的区域
class Find_interesting
{
public:
	Find_interesting(const string& config);
	void find_box(const Mat& img, vector<Rect>& box);
	CascadeClassifier cascade;
};


//三角化计算图像三维坐标（在左相机坐标系下）
void trangulation(
	const vector<KeyPoint>& keypoints_1,
	const vector<KeyPoint>& keypoints_2,
	const vector<DMatch>& matches,
	const Rectify &rectify,
	vector<Point3d>& points
);

int main(int argc, char **argv) {
	//读入参数 
    if(argc!=5)
	{
		cerr<<"usage: ./trangulate_show img_left img_right config_file detect_file"<<endl;
		return -1;
	}
	
	
	Mat img_left,img_right;
	Mat img_left_rect, img_right_rect;
	
	Mat img_show;
	
	img_left=imread(argv[1]);
	img_right=imread(argv[2]);
	
	imshow("111",img_right);
	
	waitKey();
	
	cout<<"pic load succeed"<<endl;
	
	Rectify rectify(argv[3]);
	cout<<"camera file load succeed"<<endl;
	
	//看一下矫正的效果
	rectify.doRectifyL(img_left,img_left_rect);
	rectify.doRectifyR(img_right,img_right_rect);
		
	resize(img_right_rect,img_show,Size(),0.5,0.5,CV_INTER_AREA);
	
		
	imshow("rect",img_show);
	
	//在未矫正前的图片中搜索区域
	Find_interesting detect(argv[4]);
	
	vector<Rect>object_left;
	vector<Rect>object_right;
	
	detect.find_box(img_left, object_left);
	detect.find_box(img_right,object_right);
	
	Mat search_left=img_left(object_left[0]).clone();
	Mat search_right=img_right(object_right[0]).clone();
	
	//在感兴趣的区域提取特征点
	vector<KeyPoint>keypoints_left, keypoints_right;
	Mat desciptors_left,desciptors_right;
	
	Ptr<ORB>orb=ORB::create();
	orb->detect(search_left,keypoints_left);
	orb->detect(search_right,keypoints_right);
	
	//把特征点坐标放回原来的图像像素坐标系
	for(size_t i=0; i<keypoints_left.size(); i++)
    {
        keypoints_left[i].pt = keypoints_left[i].pt + Point2f(object_left[0].x,object_left[0].y);
    }
    
    for(size_t i=0; i<keypoints_right.size(); i++)
    {
        keypoints_right[i].pt = keypoints_right[i].pt + Point2f(object_right[0].x,object_right[0].y);
    }
    
    //计算特征点的描述子
    orb->compute(img_left_rect,keypoints_left,desciptors_left);
	orb->compute(img_right_rect,keypoints_right,desciptors_right);
	
	//计算匹配，并选取较好的匹配对
	vector<DMatch> matches;
	BFMatcher matcher(NORM_HAMMING);
	matcher.match(desciptors_left,desciptors_right,matches);
	
	double max_dist=0;
    for(size_t i=0; i<desciptors_left.rows; i++)
    {
        double dist = matches[i].distance;
        if(dist>max_dist)
            max_dist=dist;
    }
    
    vector<DMatch> good_matches;
    for(size_t i=0; i<desciptors_left.rows; i++)
    {
        if(matches[i].distance <= 0.3*max_dist)
            good_matches.push_back(matches[i]);
    }
	
	//计算3d坐标
	vector<Point3d> points;
	trangulation(keypoints_left,keypoints_right,good_matches,rectify,points);
	
	for(size_t p:points)
	{
		cout<<p;
	}
	/*
	for(size_t i=0; i<object_left.size(); i++)
    {
        Rect r = object_left[i];
		rectangle(img_left_rect, r, Scalar(255,255,0),3,3,0);
    }
    
    for(size_t i=0; i<object_left.size(); i++)
    {
        Rect r = object_left[i];
		rectangle(img_right_rect, r, Scalar(255,0,255),3,3,0);
    }
    */
	
    return 0;
}

Rectify::Rectify(const string& config)
{
	FileStorage fsSetting(config, FileStorage::READ);
	if(!fsSetting.isOpened())
	{
		cerr<<"error:wrong path to setting"<<endl;
		throw;
	}
	
	cout<<"right path to setting"<<endl;
	
	Mat L2R_R,L2R_T,Q;
	//读取相机参数矩阵 K内参 D畸变 R两相机的旋转 T两相机的平移（用matlab算比较准）
	
	fsSetting["LEFT.K"]>>K_l;
	fsSetting["RIGHT.K"]>>K_r;
	
	fsSetting["LEFT.D"]>>D_l;
	fsSetting["RIGHT.D"]>>D_r;
	
	fsSetting["LEFT2RIGHT.R"]>>L2R_R;
	fsSetting["LEFT2RIGHT.T"]>>L2R_T;
	
	
	rows_l=fsSetting["LEFT.height"];
	cols_l = fsSetting["LEFT.width"];
	
	rows_r=fsSetting["RIGHT.height"];
	cols_r = fsSetting["RIGHT.width"];
	
	//由这些矩阵分别计算两相机的矫正旋转、投影矩阵
	
	stereoRectify(K_l,D_l,K_r,D_r,Size(cols_l,rows_l),L2R_R,L2R_T,R_l,R_r,P_l,P_r,Q);
	
	//计算重映射参数
	initUndistortRectifyMap(K_l,D_l,R_l,P_l.rowRange(0,3).colRange(0,3),Size(cols_l,rows_l),CV_32F,M1l,M2l);
	initUndistortRectifyMap(K_r,D_r,R_r,P_r.rowRange(0,3).colRange(0,3),Size(cols_r,rows_r),CV_32F,M1r,M2r);
	
	
}

void Rectify::doRectifyL(const Mat& img_src, Mat& img_rect)
{
	remap(img_src,img_rect,M1l,M2l,CV_INTER_LINEAR);
}

void Rectify::doRectifyR(const Mat& img_src, Mat& img_rect)
{
	remap(img_src,img_rect,M1r,M2r,CV_INTER_LINEAR);
}

Find_interesting::Find_interesting(const string& config)
{
	if(config.size()==0)
	{
		cerr<<"config file error";
		throw;
	}
	cascade.load(config);
}

void Find_interesting::find_box(const Mat& img, vector< Rect >& boxs)
{
	Mat img_gray;
	cvtColor(img,img_gray,COLOR_BGR2GRAY);
	equalizeHist(img_gray, img_gray);
	
	cascade.detectMultiScale(img_gray, boxs, 1.1, 2, 0|CASCADE_SCALE_IMAGE,Size(30,30));
	
}

Point2f pixel2cam ( const Point2d& p, const Mat& K )
{
    return Point2f
    (
        ( p.x - K.at<double>(0,2) ) / K.at<double>(0,0), 
        ( p.y - K.at<double>(1,2) ) / K.at<double>(1,1) 
    );
}

void trangulation(const vector< KeyPoint >& keypoints_1, const vector< KeyPoint >& keypoints_2, const vector< DMatch >& matches, const Rectify& rectify , vector< Point3d >& points)
{
	vector<Point2f>pts_1,pts_2;
	for(DMatch m:matches)
	{
		pts_1.push_back(pixel2cam(keypoints_1[m.queryIdx].pt,rectify.K_l));
		pts_2.push_back(pixel2cam(keypoints_2[m.trainIdx].pt,rectify.K_r));		
	}
	Mat pts_4d;
	triangulatePoints(rectify.P_l,rectify.P_r,pts_1,pts_2,pts_4d);
	
	for(int i=0;i<pts_4d.cols;i++)
	{
		Mat x = pts_4d.col(i);
		x/=x.at<float>(3,0);
		Point3d p(
			x.at<float>(0,0),x.at<float>(1,0),x.at<float>(2,0)
		);
		points.push_back(p);
	}
}
