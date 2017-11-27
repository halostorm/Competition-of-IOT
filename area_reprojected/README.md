need： opencv3.2 and opencv_contrib-3.2
camera paras can get from matlab like this: https://jingyan.baidu.com/article/22a299b5e6da909e18376a75.html
change your stereo camero paras
train your target xml

use example:

mkdir build 
cd build
cmake ..
make
cd ../bin
./area_reprojected ../pic/left.png ../pic/right.png ../config/stereo_paras.yaml ../config/haarcascade_frontalface_alt.xml
