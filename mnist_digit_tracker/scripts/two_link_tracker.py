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

class DigitTracker():

	def __init__(self):
		self.two_link_server = rospy.Service('/tracking_start', TwoLinkService, self.callback)
		self.joint_pub = rospy.Publisher('/joint_states', JointState, queue_size=1)
		self.desired_pub = rospy.Publisher('/desired_point_array', MarkerArray, queue_size=1)
		self.actual_pub = rospy.Publisher('/actual_point_array', MarkerArray, queue_size=1)
		self.circleArray = MarkerArray()
		self.markerArray = MarkerArray()
		self.delete_trajectories()
		# parameter
		self.dt = rospy.get_param("/tracker/control/dt")
		self.Kp = rospy.get_param("/tracker/control/Kp")
		self.l1 = rospy.get_param("/tracker/two_link/l1")
		self.l2 = rospy.get_param("/tracker/two_link/l2")
		self.ratio = rospy.get_param("/tracker/trajectory/ratio")
		self.offset = rospy.get_param("/tracker/trajectory/offset")
		self.r = rospy.Rate(rospy.get_param("/tracker/marker/rate"))

	def delete_trajectories(self):
		marker = Marker()
		marker.header.frame_id = "/base"
		marker.action = marker.DELETEALL
		self.circleArray.markers.append(marker)
		self.desired_pub.publish(self.circleArray)
		self.markerArray.markers.append(marker)
		self.actual_pub.publish(self.markerArray)
		#print("delete_trajectories")

	def callback(self, req):
		if req.digit == 0:
			traj = req.points
			self.draw_trajectory(traj)			
			res = True
		else:
			self.draw_initial_point()
			res = False
		return TwoLinkServiceResponse(res)

	def draw_initial_point(self):
		self.delete_trajectories()
		xd = np.array([0, self.offset])
		q = self.inverse_kinematics(xd)
		self.joint_publish(q)
		self.desired_publish(xd)
	
	def draw_trajectory(self, traj):
		i = 0
		self.delete_trajectories()
		while not rospy.is_shutdown() and i<len(traj):
			# xd: desired x, y
			# x : current x, y
			# e : tracking error
			xd = np.array([(traj[i].x-14)*self.ratio, (14-traj[i].y)*self.ratio+self.offset])
			if i == 0:
				q = self.inverse_kinematics(xd)
			x = self.forward_kinematics(q)	
			e = xd-x
			x_dot = self.Kp*e
			q_dot = np.dot(self.jacobian_inverse(q), x_dot)
			q = q + q_dot*self.dt
			i = i+1
			self.joint_publish(q)
			self.actual_publish(x)
			self.desired_publish(xd)
			self.r.sleep()

	def joint_publish(self, q):
		joint_command = JointState()	
		joint_command.header.stamp = rospy.Time.now()
		joint_command.name = ['joint1', 'joint2']
		joint_command.position = q
		self.joint_pub.publish(joint_command)
	
	def desired_publish(self, x):
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
		self.desired_pub.publish(self.circleArray)

	def actual_publish(self, x):
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
		self.actual_pub.publish(self.markerArray)

	def forward_kinematics(self, q):
		x = self.l1*math.cos(q[0]) + self.l2*math.cos(q[0]+q[1])
		y = self.l1*math.sin(q[0]) + self.l2*math.sin(q[0]+q[1])
		return np.array([x, y])

	def inverse_kinematics(self, x):
		th2 = math.pi - math.acos((self.l1*self.l1 + self.l2*self.l2 - x[0]*x[0] - x[1]*x[1])/(2*self.l1*self.l2))
		if abs(x[0]) < 1e-3:
			phi = math.pi/2.0
		else:
			phi = math.atan(x[1]/x[0])
		if phi > 0:
			th1 = phi - th2/2.0
		else:
			th1 = math.pi+phi - th2/2.0
		return np.array([th1, th2])

	def jacobian_inverse(self, q):
		J = np.array([[-self.l1*math.sin(q[0]), -self.l2*math.sin(q[0]+q[1])],
					[self.l1*math.cos(q[0]), self.l2*math.cos(q[0]+q[1])]])
		#J_inv = J.transpose()
		J_inv = inv(J)
		return J_inv


def main(args):
	rospy.init_node('two_link_tracker', anonymous=False)	
	two_link = DigitTracker()
	try:		
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting down")

if __name__ == '__main__':	
	main(sys.argv)