#include <iostream>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/tracking.hpp>

using namespace std;
using namespace cv;



int main(int argc, char **argv) {
    
    Mat frame,gray;
    
    VideoCapture cap(0);
    if(!cap.isOpened())
        return -1;
    
    //级联分类器
    CascadeClassifier cascade;
    cascade.load("haarcascade_frontalface_alt.xml");
    
    cap>>frame;
    cvtColor(frame, gray, COLOR_BGR2GRAY);
    equalizeHist(gray,gray);
    
    vector<Rect> objects;
    
    cascade.detectMultiScale(gray, objects, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30,30));
    
    Ptr<Tracker> tracker = Tracker::create("KCF");
    
    if(objects.size()>0)
        tracker->init(frame,objects[0]);
    else
        tracker->init(frame,Rect2d(1,1,1,1));
 
    Rect2d roi;
    int i=0;
    
    while(1)
    {
        
        cap>>frame;
        
        if(i==0 || i% 50 == 0 )
        {
            cvtColor(frame, gray, COLOR_BGR2GRAY);
            equalizeHist(gray,gray);
    
    
            cascade.detectMultiScale(gray, objects, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(50,50));
    
            for(size_t i=0; i<objects.size(); i++)
            {
                Rect r = objects[i];
                rectangle(frame, r, Scalar(255,255,0),3,3,0);
            }
        
            
            cout<<"reloclization  "<<objects.size()<<endl;
            
            if(objects.size()>0)
            {
                roi = objects[0];
                tracker->init(frame,roi);
            }
        }
        
        tracker->update(frame,roi);
        i++;

        
        rectangle(frame,roi,Scalar(0,255,255),2,2,0);
        
        
    
        imshow("object",frame);
    
        char c=waitKey(30);
        if(c == 27)
            break;
    

    }
    return 0;
}
