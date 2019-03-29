#include "setting.hpp"
#include "DetectProcess.hpp" 
#include <thread> 
#include <unistd.h>
#include "opencv2/highgui.hpp"
#include <iostream>

using namespace cv;
using namespace std;

/************************************************* 
    Function:       main
    Description:    function entrance
    Calls:          
                    DetectProcess::ImageProducer()
                    DetectProcess::ImageConsumer() 
    Input:          None 
    Output:         None 
    Return:         return 0
    *************************************************/
int main(int argc, char * argv[])
{
    char *config_file_name = "../param/param_config.xml";
    FileStorage fs(config_file_name, FileStorage::READ);    //initialization
    if (!fs.isOpened())     //open xml config file
    {
        std::cout << "Could not open the configuration file: param_config.xml " << std::endl;
        return -1;
    }
    Settings setting(config_file_name);

    DetectProcess image_cons_prod(&setting);
    std::thread task0(&DetectProcess::ImageReader, image_cons_prod);  // add image reading thread
    std::thread task1(&DetectProcess::ImageProcesser, image_cons_prod);  // add image processing thread

    task0.join();
    task1.join();
    
	return EXIT_SUCCESS;
}
 
