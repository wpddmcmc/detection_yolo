#include "DetectProcess.hpp"
#include "opencv2/opencv.hpp"
#include <opencv2/highgui.hpp>
#include <time.h>
#include <stdio.h> 

using namespace cv;
//Window config
#define VIDEO_WIDTH  640
#define VIDEO_HEIGHT 480

//Image buffer size
#define BUFFER_SIZE 1

volatile unsigned int prdIdx = 0;	//image reading index
volatile unsigned int csmIdx = 0;	//image processing index

//darknet variable definition
char *datacfg ;		//data path config
char *name_list ;	//class list
char **names  ;		//list name
image im;			//net input image
char *cfgfile ;		//net config file
char *weightfile;	//weight file
float thresh , hier_thresh;	//output hreshold
network *net;	//darknet network
image **alphabet;

//UI arrow
Mat leftarrow,rightarrow,sarrow,arrow;

//image processing speed output
double time_process;
char process_time[30];
char timenow[20];

//#define USE_CAMERA

struct ImageData {
	Mat img;             //camare data
	unsigned int frame;  //frame count
};

ImageData capturedata[BUFFER_SIZE];   //buffer of capture

/************************************************* 
    Function:       ImageProducer 
    Description:    Image read
    Input:          video file or camare image 
    Output:         one frame of reading iamge
    Return:         void
    Others:         none
    *************************************************/
void DetectProcess::ImageReader()
{
	Settings & setting = *_settings;
	string video_name = "../video/";

	#ifndef USE_CAMERA
	//read video file
	video_name+=setting.video_name;
	VideoCapture cap(video_name);

	#else
	//open camare
	VideoCapture cap(0);
	#endif

	if (!cap.isOpened())
    {
            std::cout<<"can't open video or cam"<<std::endl;
            return;
	}
		
	while(true)
    {
        //wait for next image
       	while(prdIdx - csmIdx >= BUFFER_SIZE);
        cap >> capturedata[prdIdx % BUFFER_SIZE].img;
        capturedata[prdIdx % BUFFER_SIZE].frame++; 	//frame is the index of picture
        ++prdIdx;
    }
}

/************************************************* 
    Function:       ImageConsumer 
    Description:    Image process
    Input:          one frame of reading iamge
    Output:         frame after processing display
    Return:         void
    Others:         none
    *************************************************/
void DetectProcess::ImageProcesser() {

	Settings & setting = *_settings;
	datacfg = "../coco.data";		//read data file, which contains（.names）file path,two (.txt)file
	name_list = option_find_str(read_data_cfg(datacfg), "names", "names.list");	//find the value of names in data file
	names = get_labels(name_list);		//get labels
	//egg 4 data code
	cfgfile = "../yolov3-tiny/yolov3-tiny.cfg";
	weightfile = "../yolov3-tiny/yolov3-tiny.weights";
	thresh = .5; hier_thresh = .5;
	net = load_network(cfgfile, weightfile, 0);		//load network
	set_batch_network(net, 1);		//set the batch of each layer 1
	alphabet = load_alphabet();		//load ASCII 32-127 in data/labels for lable displaying

	Mat frame,detect;		//frame - input image

	while(true){
		//get time to caculate processing time
		time_process = what_time_is_it_now();
		//get current time
		time_t tt;
		time(&tt);
		tt = tt + 8 * 3600; // transform the time zone
		tm *t = gmtime(&tt);
		
		sprintf(timenow, "%d-%02d-%02d-%02d:%02d:%02d",
				t->tm_year + 1900,
				t->tm_mon + 1,
				t->tm_mday,
				t->tm_hour,
				t->tm_min,
				t->tm_sec);

		while (prdIdx - csmIdx == 0);
		capturedata[csmIdx % BUFFER_SIZE].img.copyTo(frame);
		++csmIdx;
		
		imshow("frame", frame);

		if(1)
		{
			detect = frame;

			Detecter(detect);		// detect
			free_image(im);				//free GPU memory

			//FPS caculate
			float fps = 1/(what_time_is_it_now() - time_process);
			sprintf(process_time,"FPS: %.2f ",fps);
			putText(detect,process_time,Point(15,20),CV_FONT_HERSHEY_SIMPLEX , 0.8, Scalar(0, 0, 0), 1);
			putText(detect,timenow,Point(1000,20),CV_FONT_HERSHEY_SCRIPT_COMPLEX  , 0.5, Scalar(0, 255, 128), 1);
			imshow("detect", detect);
		}	
		
		if(setting.debug_mode<1)	//wait for keyboard press
		{
			char key = waitKey(5);
			if(key == 27)
				exit(0);
			if(key == 13)
			{
				char filename[40];
				sprintf(filename,"../output/%s.jpg",timenow);
				cout<<"Note: "<<filename<<" write to file sucess!";
				//imwrite(filename,road);
			}
		}
		else	//cotinue processing
		{
			char key = waitKey(0);
			if(key == 27)
				exit(0);
			if(key == 13)
			{
				char filename[40];
				sprintf(filename,"../output/%s.jpg",timenow);
				cout<<"Note: "<<filename<<" write to file sucess!";
				imwrite(filename,detect);
			}
			
		}
	}
}

/************************************************* 
    Function:       Mat2Image 
    Description:   	change format: mat->image
	Input:          
					Mat RefImg - mat image need to reformat
					image *im - the output image format image
    Output:         image *im - the output image format image
    Return:         void
    Others:         none
    *************************************************/
void DetectProcess::Mat2Image(Mat RefImg,image *im)
{
	CV_Assert(RefImg.depth() == CV_8U);		//judge if  RefImag is CV_8U
	int h = RefImg.rows;
	int w = RefImg.cols;
	int channels = RefImg.channels();
	*im = make_image(w, h, 3);		//create 3 channels image
	int count = 0;
	switch (channels)
	{
	case 1:
	{
		MatIterator_<unsigned char> it, end;
		for (it = RefImg.begin<unsigned char>(), end = RefImg.end<unsigned char>(); it != end; ++it)
		{
			im->data[count] = im->data[w * h + count] = im->data[w * h * 2 + count] = (float)(*it) / 255.0;

			++count;
		}
		break;
	}

	case 3:
	{
		MatIterator_<Vec3b> it, end;
		for (it = RefImg.begin<Vec3b>(), end = RefImg.end<Vec3b>(); it != end; ++it)
		{
			im->data[count] = (float)(*it)[2] / 255.0;
			im->data[w * h + count] = (float)(*it)[1] / 255.0;
			im->data[w * h * 2 + count] = (float)(*it)[0] / 255.0;

			++count;
		}
		break;
	}

	default:
		printf("Channel number not supported.\n");
		break;
	}
}

/************************************************* 
    Function:       get_pixel 
    Description:   	change format: image->mat
	Input:          
					image m	- image need to get pixel
					int x - width
					int y - height
					int c - channels
    Output:         pixel of input image
    Return:         float
    Others:         none
    *************************************************/
float get_pixel(image m, int x, int y, int c)
{
    assert(x < m.w && y < m.h && c < m.c);
    return m.data[c*m.h*m.w + y*m.w + x];
}

/************************************************* 
    Function:       image2mat 
    Description:   	change format: image->mat
	Input:          
					image p	- image image need to reformat
					Mat *Img -	the output mat format image
    Output:         Mat *Img - the output immatage format image
    Return:         void
    Others:         none
    *************************************************/
void DetectProcess::Image2Mat(image p,Mat &Img)
{
	IplImage *disp = cvCreateImage(cvSize(p.w,p.h), IPL_DEPTH_8U, p.c);
    image copy = copy_image(p);
    constrain_image(copy);

	int x, y, k;
	if (p.c == 3)
		rgbgr_image(p);

	int step = disp->widthStep;
	for (y = 0; y < p.h; ++y)
	{
		for (x = 0; x < p.w; ++x)
		{
			for (k = 0; k < p.c; ++k)
			{
				disp->imageData[y * step + x * p.c + k] = (unsigned char)(get_pixel(p, x, y, k) * 255);
			}
		}
	}
	if (0)
	{
		int w = 448;
		int h = w * p.h / p.w;
		if (h > 1000)
		{
			h = 1000;
			w = h * p.w / p.h;
		}
		IplImage *buffer = disp;
		disp = cvCreateImage(cvSize(w, h), buffer->depth, buffer->nChannels);
		cvResize(buffer, disp, CV_INTER_LINEAR);
		cvReleaseImage(&buffer);
	}
	
	Img=cvarrToMat(disp);
	free_image(copy);
   	cvReleaseImage(&disp);
}

/************************************************* 
    Function:       detecter 
    Description:   	darknet detect objects
	Input:          
					Mat &src - image need to detect
    Output:         Mat &src - image after drawing detect target
    Return:         void
    Others:         none
    *************************************************/
void DetectProcess::Detecter(Mat &src)
{  
	//format change
	Mat2Image(src,&im);
    float nms=.45;
   	
    layer l = net->layers[(net->n)-1];    
    image sized = letterbox_image(im, net->w, net->h);
    float *X = sized.data;
	network_predict(net, X);

	int nboxes = 0;
	detection *dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes);
	if (nms)
	{
		do_nms_sort(dets, nboxes, 80, nms);
	}
	int rect_scalar[nboxes][4];	//rect_scalar[i][0] left rect_scalar[i][1] right rect_scalar[i][2] top rect_scalar[i][3] bottom
	//draw_detections(im, dets, nboxes, thresh, names, alphabet,80);
	get_detections(im, dets, nboxes, thresh, names, alphabet,80,rect_scalar);
	vector<Rect> detectBox;
	for(int i=0;i<nboxes;i++)
	{
		detectBox.push_back(Rect(rect_scalar[i][0],rect_scalar[i][2],rect_scalar[i][1]-rect_scalar[i][0],rect_scalar[i][3]-rect_scalar[i][2]));
	}
	Mat result;
	Image2Mat(im,result);
	for(int i=0;i<detectBox.size();i++)
	{
		char position[50];
		sprintf(position,"(%d,%d)[%d,%d]",detectBox[i].x,detectBox[i].y,detectBox[i].width,detectBox[i].height);
		putText(result,position,Point(detectBox[i].tl().x,detectBox[i].tl().y+30),CV_FONT_HERSHEY_PLAIN, 1, Scalar(128, 0, 255), 1);
	}
	result.copyTo(src);
    free_detections(dets, nboxes);
	free_image(sized);
}
