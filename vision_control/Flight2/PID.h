#include <math.h>
#include <iostream>

class PIDctrl
{
	public:
	double Kp;  //比例系数
	double Ki;   //积分系数
	double Kd;  //微分系数
	double preErr;    //前一拍误差
	double sumErr;    //误差累积
	double dErr;
	double output;
	double outMax;//输出最大值
	double sumMax;
	
	PIDctrl(double Kp,double Ki,double Kd,double max)
	{
		this->Kp=Kp;
		this->Ki=Ki;
		this->Kd=Kd;
		outMax=max;
		preErr=0.0;sumErr=0.0;sumMax=50000;
	};
	
	void calc(double &curErr);
	void reset();
};



void PIDctrl::reset()
{
	preErr=0.0;
	sumErr=0.0; 
	dErr=0.0;
}



void PIDctrl::calc(double &curErr)
{
	if(fabs(sumErr+curErr)<sumMax) sumErr+=curErr;
	dErr=curErr-preErr;
	preErr=curErr;
	output=Kp*curErr+Ki*sumErr+Kd*dErr;
	if (abs(output)>outMax)
	{
		output=outMax*output/abs(output);
	}
}
