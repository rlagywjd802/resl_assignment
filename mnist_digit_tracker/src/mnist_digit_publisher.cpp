#include <ros/ros.h>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <image_transport/image_transport.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <signal.h>
#include <termios.h>
#include <cstdlib>
#include <ctime>

#define KEYCODE_0 0x30 
#define KEYCODE_1 0x31
#define KEYCODE_2 0x32
#define KEYCODE_3 0x33
#define KEYCODE_4 0x34
#define KEYCODE_5 0x35 
#define KEYCODE_6 0x36
#define KEYCODE_7 0x37
#define KEYCODE_8 0x38
#define KEYCODE_9 0x39

using namespace std;

int kfd = 0;
struct termios cooked, raw;

void quit(int sig)
{
	(void)sig;
  	tcsetattr(kfd, TCSANOW, &cooked);
	ros::shutdown();
	exit(0);
}

class DigitPublisher
{
private:
	ros::NodeHandle nh;
	image_transport::ImageTransport it;
	image_transport::Publisher image_pub;
	vector<cv::Mat> test_images;
	vector<double> test_labels;
	vector< vector<cv::Mat> > test_images_sorted;
public:
	DigitPublisher();
	void key_loop();
	void digit_publish(int num);
	int convert_to_int(int i);
	void load_mnist_img(vector<cv::Mat> &vec);
	void load_mnist_lbl(vector<double> &vec);
	void sort_mnist_img(vector<vector<cv::Mat> > &vec);
	int get_random_number(int num);
};

DigitPublisher::DigitPublisher(): it(nh), test_images(), test_labels()
{
	std::string topic_name;
	int queue_size;
	nh.param("/publisher/topic", topic_name, std::string("/digit_image"));
	nh.param("/publisher/queue_size", queue_size)
	image_pub = it.advertise(topic_name, queue_size);
	//image_pub = it.advertise("/digit_image", 1);
	load_mnist_img(test_images);
	load_mnist_lbl(test_labels);
	sort_mnist_img(test_images_sorted);
}

void DigitPublisher::key_loop()
{
	char c;

	// get the console in raw mode                                                              
	tcgetattr(kfd, &cooked);
	memcpy(&raw, &cooked, sizeof(struct termios));
	raw.c_lflag &=~ (ICANON | ECHO);
	// Setting a new line, then end of file                         
	raw.c_cc[VEOL] = 1;
	raw.c_cc[VEOF] = 2;
	tcsetattr(kfd, TCSANOW, &raw);

	puts("Reading from keyboard");
	puts("---------------------------");
	puts("Press 0...9 key from keyboard to publish an image of a digit.");

	for(;;)
	{
		// get the next event from the keyboard  
		if(read(kfd, &c, 1) < 0)
		{
		  perror("read():");
		  exit(-1);
		}

		ROS_DEBUG("value: 0x%02X", c);

		switch(c)
		{
			case KEYCODE_0:
				ROS_INFO("You pressed 0");
				digit_publish(0);
				break;

			case KEYCODE_1:
				ROS_INFO("You pressed 1");
				digit_publish(1);
				break;

			case KEYCODE_2:
				ROS_INFO("You pressed 2");
				digit_publish(2);
				break;

			case KEYCODE_3:
				ROS_INFO("You pressed 3");
				digit_publish(3);
				break;

			case KEYCODE_4:
				ROS_INFO("You pressed 4");
				digit_publish(4);
				break;

			case KEYCODE_5:
				ROS_INFO("You pressed 5");
				digit_publish(5);
				break;

			case KEYCODE_6:
				ROS_INFO("You pressed 6");
				digit_publish(6);
				break;

			case KEYCODE_7:
				ROS_INFO("You pressed 7");
				digit_publish(7);
				break;

			case KEYCODE_8:
				ROS_INFO("You pressed 8");
				digit_publish(8);
				break;

			case KEYCODE_9:
				ROS_INFO("You pressed 9");
				digit_publish(9);
				break;
				
		}	   

	}

	return;
}

void DigitPublisher::digit_publish(int num)
{
	int r_num = get_random_number(num);
	cv_bridge::CvImagePtr cv_ptr(new cv_bridge::CvImage);

	ros::Time time = ros::Time::now();
	cv_ptr->encoding = "mono8";	//CV_8UC1, grayscale image
	cv_ptr->header.stamp = time;
	cv_ptr->header.frame_id = "/digit_image";
	cv_ptr->image = test_images_sorted[num][r_num];
	image_pub.publish(cv_ptr->toImageMsg());
}

int DigitPublisher::convert_to_int(int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;

    return((int) ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void DigitPublisher::load_mnist_img(vector<cv::Mat> &vec)
{
	ifstream file ("/home/hyojeong/catkin_ws/src/mnist_digit_tracker/dataset/t10k-images-idx3-ubyte", ios::binary);
	int magic_number = 0;
	int number_of_images = 0;
	int n_rows = 0;
	int n_cols = 0;

	if(file.is_open())
	{
		// read mnist info
		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = convert_to_int(magic_number);
		file.read((char*)&number_of_images, sizeof(number_of_images));
		number_of_images = convert_to_int(number_of_images);
		file.read((char*)&n_rows, sizeof(n_rows));
		n_rows = convert_to_int(n_rows);
		file.read((char*)&n_cols, sizeof(n_cols));
		n_cols = convert_to_int(n_cols);

		// read mnist image
		for(int i=0; i<number_of_images; ++i)
		{
			cv::Mat tp = cv::Mat::zeros(n_rows, n_cols, CV_8UC1);
			for(int r=0; r<n_rows; ++r)
			{
				for(int c=0; c<n_cols; ++c)
				{
					unsigned char temp = 0;
					file.read((char *)&temp, sizeof(temp));
					tp.at<uchar>(r, c) = (int)temp;
				}
			}
			vec.push_back(tp);
		}


	}
	else
	{
		ROS_INFO("cannot open file");
	}	
}

void DigitPublisher::load_mnist_lbl(vector<double> &vec)
{
	ifstream file ("/home/hyojeong/catkin_ws/src/mnist_digit_tracker/dataset/t10k-labels-idx1-ubyte", ios::binary);
	int magic_number = 0;
	int number_of_images = 0;

	if(file.is_open())
	{
		// read mnist info
		file.read((char*) &magic_number, sizeof(magic_number));
        magic_number = convert_to_int(magic_number);
        file.read((char*) &number_of_images,sizeof(number_of_images));
        number_of_images = convert_to_int(number_of_images);

        // read mnist label
        for(int i = 0; i < number_of_images; ++i)
        {
            unsigned char temp = 0;
            file.read((char*) &temp, sizeof(temp));
            vec.push_back((double)temp);
        }
	}
	else
	{
		ROS_INFO("cannot open file");
	}
}

void DigitPublisher::sort_mnist_img(vector<vector <cv::Mat> > &vec)
{
	for(int i=0; i<=9; i++)
	{
		vector<cv::Mat> temp;
		for(int j=0; j<test_labels.size(); j++)
		{
			if(test_labels[j] == i)
			{
				temp.push_back(test_images[j]);	
			}
		}
		vec.push_back(temp);
	}	
}

int DigitPublisher::get_random_number(int num)
{
	return rand()%(test_images_sorted[num].size());
}


int main(int argc, char **argv)
{
	srand(time(NULL));
	ros::init(argc, argv, "mnist_digit_publisher");
	DigitPublisher image_publisher;

	signal(SIGINT, quit);

	image_publisher.key_loop();

	return 0;
}
