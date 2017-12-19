#include"my_refind/config.h"

Config::Config(const string& camerapara)
{
  //读入文件
	FileStorage fsSetting(camerapara, FileStorage::READ);
	if(!fsSetting.isOpened())
	{
		cerr<<"error:wrong path to setting"<<endl;
		throw;
	}
	cout<<"right path to setting"<<endl;
	
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
	
	//即使重映射矩阵
	stereoRectify(K_l,D_l,K_r,D_r,Size(cols_l,rows_l),L2R_R,L2R_T,R_l,R_r,P_l,P_r,Q);
	//计算重映射参数
	initUndistortRectifyMap(K_l,D_l,R_l,P_l.rowRange(0,3).colRange(0,3),Size(cols_l,rows_l),CV_32F,M1l,M2l);
	initUndistortRectifyMap(K_r,D_r,R_r,P_r.rowRange(0,3).colRange(0,3),Size(cols_r,rows_r),CV_32F,M1r,M2r);

}

void Config::doRectifyL(const Mat& img_src, Mat& img_rect)
{
  //矫正图像
	remap(img_src,img_rect,M1l,M2l,CV_INTER_LINEAR);
}

void Config::doRectifyR(const Mat& img_src, Mat& img_rect)
{
	remap(img_src,img_rect,M1r,M2r,CV_INTER_LINEAR);
}

