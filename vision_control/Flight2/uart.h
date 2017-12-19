#include     <stdio.h>      /*标准输入输出定义*/
#include     <stdlib.h>     /*标准函数库定义*/
#include     <unistd.h>     /*Unix 标准函数定义*/
#include     <sys/types.h>  
#include     <sys/stat.h>   
#include     <fcntl.h>      /*文件控制定义*/
#include     <termios.h>    /*PPSIX 终端控制定义*/
#include     <errno.h>      /*错误号定义*/


int uartOpen(const char * dev_name)
{
	int fd;
    fd = open(dev_name, O_RDWR | O_NOCTTY | O_NDELAY);
    if(fd == -1)
    {
        perror("open_port: Unable to open. \n");
    }
    else  
    {
        fcntl(fd, F_SETFL, 0);
        printf("port %s is open.\n",dev_name);
    }
    return(fd);
}


int uartSet(int fd){                      
    struct termios opt;              
    cfsetispeed(&opt, B9600);      
    cfsetospeed(&opt, B9600);
	//8数据 1停止 无校验
    opt.c_cflag &= ~PARENB;          
    opt.c_cflag &= ~CSTOPB;
    opt.c_cflag &= ~CSIZE;
    opt.c_cflag |= CS8;

   if ( tcsetattr(fd, TCSANOW, &opt)!=0)
		perror("set error!\nS");   
	tcflush(fd,TCIFLUSH);  
    return(fd);
}


int uartSenddxdy(int fd, int dx,int dy)
{
	unsigned char data_send[]={0x7F,0x00,0x00,0x00,0x00,0x7E};
	short int *data_ptr=(short *)(data_send+1);
	data_ptr[0]=(short)(dx);
	data_ptr[1]=(short)(dy);
	printf("%2X %2X %2X %2X %2X %2X",data_send[0],data_send[1],data_send[2],data_send[3],data_send[4],data_send[5]);
	if(write(fd, data_send, 6)!=6)
		return -1;
	else{
		return 0;
	}
}

int uartSend(int fd, unsigned char *data,int len)
{
	if(write(fd, data, len)!=len)
		return -1;
	else{
		return 0;
	}
}


