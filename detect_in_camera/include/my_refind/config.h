#ifndef CONFIG_H
#define CONFIG_H

#include "my_refind/common.h"

//相机参数和矫正的类
class Config
{
public:
  
  //类构造时带名称
	Config(const string& camerapara);
	
	//左右相机的相机参数和畸变参数
	Mat K_l,K_r,D_l,D_r;
	
	//左相机坐标系到右相机坐标系的旋转和平移
	Mat L2R_R,L2R_T;
	
	//左右相机的高和宽
	int rows_l,cols_l,rows_r,cols_r;
	//重投影矩阵
	Mat P_l,P_r,R_l,R_r;
	//X，Y的重映射参数
	Mat M1l,M1r,M2l,M2r;
	
	//深度视差映射矩阵
	Mat Q;
	
	//对左右相机进行矫正
	void doRectifyL(const Mat& img_src, Mat& img_rect);
	void doRectifyR(const Mat& img_src, Mat& img_rect);
	
};

#endif 