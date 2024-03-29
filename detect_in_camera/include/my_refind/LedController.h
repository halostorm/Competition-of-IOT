#pragma once
#include <unistd.h>
#include <fcntl.h>
bool led_status;
static void *ledloop(void *data){
	int fd2led;
	if(!(fd2led = open("/sys/class/gpio/gpio16/value", O_WRONLY))){cout<<"cannot open gpio"<<endl;return 0;}
	while(1){
		if(led_status){
			write(fd2led, "0", strlen("1"));
			//cout<<"print1"<<endl;
		}
		else {
			write(fd2led, "1", strlen("1"));
			//cout<<"print0"<<endl;
		}
	}	
}
class LedController {
public:
    LedController(){
        if (pthread_create(&led_thread, NULL, ledloop, NULL) != 0)
	{
		printf("led_thread create");
	}
	pthread_detach(led_thread);
	
    }
    void ledON(){
        led_status=true;
    }
    void ledOFF(){
        led_status=false;
    }
    void flash(int time){
        if (time > flash_cnt){
            led_status = !led_status;
            flash_cnt = 0;
        }
        if (led_status)
            ledON();
        else
            ledOFF();
        ++flash_cnt;
    }
	int readled(){
		int fd2ledin=open("/sys/class/gpio/gpio16/value",O_RDONLY);
		char value;
		read(fd2ledin, &value,1);
		//printf("-----%c------",value);
		close(fd2ledin);
		if (value=='1')
			return 1;
		else
			return 0;
	}

public:
    int flash_cnt;

    pthread_t led_thread;
};
