# detection_yolo

---

Mingcong Chen 27/3/2019

---

### 1.Introduction

object detecting

### 2.Environment

**System Dependency:** Ubuntu16.04

** Software Dependency:** Cmake2.8 or upper, C++11, OpenCV3.4.0(only), darknet

** Hardware Environment:** 

```
Computer1: CPU:i7-4720HQ GPU:Nvidia GTX960M-2GB RAM:12GB
Computer2: CPU:i7-8750H GPU:Nvidia GTX1060-6GB RAM:8GB
```

### 3.File Structure

**src:** source code file
**include: **head file
**build: ** compile file
**build/data: ** yolo data list file
**yolov3-tiny: **yolo network file
**video: **test video
### 4.Installation

#### i. Make sure your Software Dependency has been installed

#### ii.Configure compile

Configure ```CMakeLists.txt``` .

```bash
gedit CMakeLists.txt
```

if you have OpenCV version more than one, uncommit and change line8,set the build path which one you want to use.

```cmake
#if u have OpenCV version more than one, set the build path which one u want to use
set(OpenCV_DIR "YOUR_PATH")
```

Ex:

```cmake
#if u have OpenCV version more than one, set the build path which one u want to use
set(OpenCV_DIR "/home/test/app/opencv-3.4.0/build/")
```

In line50,51, configure the darknet path and cuda path in include_directories

```cmake
include_directories (
    ${OpenCV_INCLUDE_DIRS}
    /usr/local/include
    /usr/include 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
#darknet path
    YOUR_PATH/include
    YOUR_PATH/src
#cuda path
    YOUR_PATH/include    
)
```

Ex:

```cmake
include_directories (
    ${OpenCV_INCLUDE_DIRS}
    /usr/local/include
    /usr/include 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
#darknet path
    /home/test/app/darknet/include
    /home/test/app/darknet/src
#cuda path
    /usr/local/cuda-9.0/include    
)
```

In line62, configure the darknet library path in target_link_libraries

```cmake
target_link_libraries(autocar
    ${OpenCV_LIBS}
    /usr/lib
    /usr/local/lib
    ${DEPENDENCIES}
    #darknet lib path
    YOUR_PATH/libdarknet.so
)
```

Ex:

```cmake
target_link_libraries(autocar
    ${OpenCV_LIBS}
    /usr/lib
    /usr/local/lib
    ${DEPENDENCIES}
    #darknet lib path
    /home/test/app/darknet/libdarknet.so
)
```

#### iii.Compile

Entry ```build/``` and compile

```bash
cd build
cmake ..
make
```

#### iv.Run

```bash
./autocar
```

### 5.Using Note

#### i.Code change

if you changed the code , then ***need to ```make```to compile again***.

#### ii.Code file change

If any file changes(add, delete, rename), you**need to ```cmake ..``` and ```make```in ```build/```**

#### iii. Camera using

In source file ```ImageConsProd.cpp``` line31, uncommit ```#define USE_CAMERA```. Then you can use camera (default 0, you can chage camera index at line49  ```VideoCapture cap(0);```), after that you **need to ``make`` to compile**

#### iv.Video file using

In source file ``` ImageConsProd.cpp``` line31, commit ```#define USE_CAMERA```. Then you can use video file, after that you **need to ``make`` to compile**

The choosing of video file. Put video file into ```video/``` folder, then change the video name you want to use in line4 of ```param_config.xml``` in ```param/```. **Note: just save change, no need to re-compile**

#### v.Debug Mode

In Debug Mode, detect keyboard after load video or camera image. Each frame load with each click of keyboard.

Debug Mode ON/OFF

Change line3 of ```param_config.xml``` in ```param/```, 1 ON, 0 OFF.

```xml
<?xml version="1.0"?>
<opencv_storage>
<debug_mode>1</debug_mode>
<video_name>vedioname.mp4</video_name>
</opencv_storage>
```

