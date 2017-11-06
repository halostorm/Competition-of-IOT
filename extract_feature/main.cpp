#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/features2d.hpp>

using namespace std;
using namespace cv;




int main(int argc, char **argv) {
    
    //读取图片
    Mat img_left,img_right;
    
    string left = argv[1];
    string right = argv[2];
    
    img_left = imread(left, CV_LOAD_IMAGE_ANYCOLOR);
    img_right = imread(right, CV_LOAD_IMAGE_ANYCOLOR);
    
    imshow("img_left",img_left);
    imshow("img_right",img_right);
    
    waitKey();
    
    //加载分类器
    CascadeClassifier cascade;
    cascade.load("haarcascade_frontalface_alt.xml");
    
    CascadeClassifier cascade1;
    cascade1.load("haarcascade_frontalface_alt.xml");

    
    Mat gray_left, gray_right;
    
    
    //灰度化 直方图均衡
    cvtColor(img_left, gray_left, COLOR_BGR2GRAY);
    equalizeHist(gray_left,gray_left);
    
    cvtColor(img_right, gray_right, COLOR_BGR2GRAY);
    equalizeHist(gray_right,gray_right);
    
    cout<<"gray succeed"<<endl;
    

    //寻找目标
    vector<Rect> objects_left;
    cascade.detectMultiScale(gray_left, objects_left, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30,30));
    
    
    vector<Rect> objects_right;
    cascade1.detectMultiScale(gray_right, objects_right, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30,30));
    
    cout<<"detect succeed"<<endl;
    
    //矩阵框出来
    for(size_t i=0; i<objects_left.size(); i++)
    {
        Rect r = objects_left[i];
        rectangle(img_left, r, Scalar(255,255,0),3,3,0);
    }
    
    for(size_t i=0; i<objects_right.size(); i++)
    {
        
        Rect r = objects_right[i];
        rectangle(img_right, r, Scalar(255,0,255),3,3,0);
    }
    
    //搜索窗
    Mat search_left = img_left(objects_left[0]).clone();
    Mat search_right = img_right(objects_right[0]).clone();
    
    
    //提取特征点
    vector<KeyPoint> keypoint_left, keypoint_right;
    Mat desciptors_left, desciptors_right;
    
    Ptr<ORB> orb = ORB::create();
    orb->detect(search_left,keypoint_left);
    orb->detect(search_right,keypoint_right);
    
    //特征点位置放到原图
    for(size_t i=0; i<keypoint_left.size(); i++)
    {
        keypoint_left[i].pt = keypoint_left[i].pt + Point2f(objects_left[0].x,objects_left[0].y);
    }
    
    for(size_t i=0; i<keypoint_right.size(); i++)
    {
        keypoint_right[i].pt = keypoint_right[i].pt + Point2f(objects_right[0].x,objects_right[0].y);
    }
    
    //计算描述子
    orb->compute(img_left,keypoint_left,desciptors_left);
    orb->compute(img_right,keypoint_right,desciptors_right);
    
    
    //计算匹配
    vector<DMatch> matches;
    BFMatcher matcher (NORM_HAMMING);
    matcher.match(desciptors_left,desciptors_right,matches);
    
    //选取较好的匹配，这一块是特征里面一个难点，我这就临时用最简单的方法，有其他开源算法
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
    
    vector<Point3d>point_3d;
    
    //z=fb/d d=ux-uy 这就测试一下
    for(size_t i=0; i<good_matches.size(); i++)
    {
        Point3d p(
            keypoint_left[good_matches[i].queryIdx].pt.x,
            keypoint_left[good_matches[i].queryIdx].pt.y,
            420*10/(keypoint_left[good_matches[i].queryIdx].pt.x-keypoint_right[good_matches[i].trainIdx].pt.x)
            
        );
        point_3d.push_back(p);
    }
    
    //画出来
    Mat matches_show;
    drawMatches(img_left,keypoint_left,img_right,keypoint_right,matches,matches_show);
    imshow("show_matches",matches_show);
    
    Mat good_show;
    drawMatches(img_left,keypoint_left,img_right,keypoint_right,good_matches,good_show);
    imshow("good_matches",good_show);
    
    

    for(Point3d p:point_3d)
    {
        cout<<p.x<<" "<<p.y<<" "<<p.z<<endl;
    }

    
    
    
    /*
    
    Mat img(img_left.rows,img_left.cols*2,CV_8UC3);
    
    Mat img_part1 = img(Rect(0,0,img_left.cols,img_left.rows));
    
    Mat img_part2 = img(Rect(img_left.cols,0,img_left.cols,img_left.rows));
    
    resize(img_left,img_part1,img_part1.size(),0,0,CV_INTER_AREA);
    resize(img_right,img_part2,img_part2.size(),0,0,CV_INTER_AREA);
    
    imshow("two pic", img);
    
    */
    
    waitKey();
    
    
    
    
    
    return 0;
}
