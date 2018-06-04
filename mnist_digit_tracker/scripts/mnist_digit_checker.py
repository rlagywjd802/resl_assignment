#! /usr/bin/env python
import rospy
import sys, os
import cv2
#import time
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from lib.mnist import load_mnist
from lib.network import TwoLayerNet
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from mnist_digit_tracker.srv import *

# Just disables the warning, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

package_path = "/home/hyojeong/catkin_ws/src/mnist_digit_tracker/"

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
network.restore_network(package_path)

class image_converter():

    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/digit_image", Image, self.callback)
        self.image_pub = rospy.Publisher("/digit_image_contour", Image, queue_size = 1)
        self.threshold = rospy.get_param("/tracker/image/threshold")

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "mono8")
            cv_image_flatten = cv_image.reshape(-1, 784)
            with network.session.as_default():
                digit_result = np.argmax(network.predict(cv_image_flatten))
                rospy.loginfo("I think it is "+ str(digit_result))
        except CvBridgeError as e:
            print(e)
        
        # Find contours
        ret,thresh = cv2.threshold(cv_image,self.threshold,255,0)
        cv_image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        
        # Find the index of the largest countour
        size = [len(c) for c in contours]
        max_index = np.argmax(size)

        # Get trajectory from the largest contour
        point_array = []
        for pA in contours[max_index]:
            p = Point()
            p.x = pA[max_index][0]
            p.y = pA[max_index][1]
            p.z = 0
            point_array.append(p)
        
        # Draw largest Contour
        cv_image_color = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB)
        cv2.drawContours(cv_image_color, contours, max_index, (0, 255, 0), 1)
        
        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image_color, encoding="bgr8"))
            if self.service_client(digit_result, point_array):
                rospy.loginfo("Finished tracking the digit '0'")
            else:
                rospy.loginfo("Parked in the middle")
        except CvBridgeError as e:
          print(e)

    def service_client(self, dig, pA):
        rospy.wait_for_service('/tracking_start')
        try:
            ros_service = rospy.ServiceProxy('/tracking_start', TwoLinkService)
            res = ros_service(dig, pA)
            return res.end
        except rospy.ServiceException, e:
            print("Failed to call tracking_start_server")


def main(args):
    rospy.init_node('image_converter', anonymous=True)
    ic = image_converter()    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == '__main__':
    main(sys.argv)
