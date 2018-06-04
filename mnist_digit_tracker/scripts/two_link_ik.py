#! /usr/bin/env python
import rospy
import math
import sys
import numpy as np
from numpy.linalg import inv
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Point
from mnist_digit_tracker.srv import *

MARKERS_MAX = 20

class DigitTracker():

	def __init__(self):
		self.two_link_server = rospy.Service('/tracking_start', TwoLinkService, self.callback)
		self.joint_pub = rospy.Publisher('/joint_states', JointState, queue_size=1)
		self.circle_pub = rospy.Publisher('/desired_point_array', MarkerArray, queue_size=1)
		self.marker_pub = rospy.Publisher('/actual_point_array', MarkerArray, queue_size=1)
		self.circleArray = MarkerArray()
		self.markerArray = MarkerArray()
		self.initialize()
		# parameter
		self.dt = 0.01
		self.q0 = [0.93, 1.29]
		self.Kp = 5
		self.l1 = 0.5
		self.l2 = 0.5
		self.r = rospy.Rate(1/self.dt)
		self.ratio = 0.1

	def initialize(self):
		marker = Marker()
		marker.header.frame_id = "/base"
		marker.action = marker.DELETEALL
		self.circleArray.markers.append(marker)
		self.circle_pub.publish(self.circleArray)
		self.markerArray.markers.append(marker)
		self.marker_pub.publish(self.markerArray)
		print("Initialized")

	def callback(self, req):
		if req.digit == 0:
			traj = req.points
			res = True
			self.talker(traj)
		else:
			res = False
		return TwoLinkServiceResponse(res)

	def talker(self, traj):
		i = 0
		q = self.q0
		sorted_traj = self.sort_trajectory(traj)
		self.initialize()
		while not rospy.is_shutdown() and i<100*2*math.pi:
			# xd: desired x, y
			# x : current x, y
			# e : tracking error
			xd = np.array([0.2*math.sin(i/100.0), 0.2*math.cos(i/100.0)+0.6])
			x = self.forward_kinematics(q)
			e = xd-x
			x_dot = self.Kp*e
			q_dot = np.dot(self.jacobian_inverse(q), x_dot)
			q = q + q_dot*self.dt
			self.joint_publish(i, q)
			if i%10 == 0:
				self.circle_publish(xd)
				self.marker_publish(x)
			i = i+1
			self.r.sleep()
	
	def sort_trajectory(self, traj):
		pA = []
		for i in range(len(traj)):
			pA.append([traj[i].x, traj[i].y])
		pA.sort()
		print(pA)

	def joint_publish(self, i, q):
		joint_command = JointState()
		#joint_command.header.seq = i		
		joint_command.header.stamp = rospy.Time.now()
		joint_command.name = ['joint1', 'joint2']
		joint_command.position = q
		self.joint_pub.publish(joint_command)
	
	def circle_publish(self, x):
		marker = Marker()
		marker.header.frame_id = "/base"
		marker.type = marker.SPHERE
		marker.action = marker.ADD
		marker.pose.position.x = x[0]
		marker.pose.position.y = x[1]
		marker.pose.position.z = 0
		marker.pose.orientation.x = 0.0
		marker.pose.orientation.y = 0.0
		marker.pose.orientation.z = 0.0
		marker.pose.orientation.w = 1.0
		marker.scale.x = 0.01
		marker.scale.y = 0.01
		marker.scale.z = 0.01
		marker.color.a = 1.0
		marker.color.r = 1.0
		marker.color.g = 0.0
		marker.color.b = 0.0
		self.circleArray.markers.append(marker)
		id = 0
		for m in self.circleArray.markers:
			m.id = id
			id += 1
		self.circle_pub.publish(self.circleArray)

	def marker_publish(self, x):
		marker = Marker()
		marker.header.frame_id = "/base"
		marker.type = marker.SPHERE
		marker.action = marker.ADD
		marker.pose.position.x = x[0]
		marker.pose.position.y = x[1]
		marker.pose.position.z = 0
		marker.pose.orientation.x = 0.0
		marker.pose.orientation.y = 0.0
		marker.pose.orientation.z = 0.0
		marker.pose.orientation.w = 1.0
		marker.scale.x = 0.01
		marker.scale.y = 0.01
		marker.scale.z = 0.01
		marker.color.a = 1.0
		marker.color.r = 0.0
		marker.color.g = 1.0
		marker.color.b = 0.0
		self.markerArray.markers.append(marker)
		id = 0
		for m in self.markerArray.markers:
			m.id = id
			id += 1
		self.marker_pub.publish(self.markerArray)

	def forward_kinematics(self, q):
		x = self.l1*math.cos(q[0]) + self.l2*math.cos(q[0]+q[1])
		y = self.l1*math.sin(q[0]) + self.l2*math.sin(q[0]+q[1])
		return np.array([x, y])

	def jacobian_inverse(self, q):
		J = np.array([[-self.l1*math.sin(q[0]), -self.l2*math.sin(q[0]+q[1])],
					[self.l1*math.cos(q[0]), self.l2*math.cos(q[0]+q[1])]])
		#J_inv = J.transpose()
		J_inv = inv(J)
		return J_inv


def main(args):
	rospy.init_node('two_link_ik', anonymous=False)	
	two_link = DigitTracker()
	try:		
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting down")

if __name__ == '__main__':	
	main(sys.argv)
