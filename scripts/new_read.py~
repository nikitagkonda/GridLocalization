#!/usr/bin/env python
import rospy
import rosbag
import rospkg
import numpy as np
import tf
import scipy.stats as sp
from lab4.msg import Motion
from lab4.msg import Observation
from geometry_msgs.msg import Point
import math
from visualization_msgs.msg import Marker

def getAbsLoc(a):
	return float((a*20)+10)

def getGridLoc(a):
	a=int(a/20)
	if a>=35:
		a=34
	if a<=-1:
		a=0
	return a
	
def getGridTheta(a):
	return int(a/10)

def getAbsTheta(a):
	return float(a*10+5)

def getTagPos(tag):
	if tag==0:
		x=125
		y=525
	elif tag==1:
		x=125
		y=325
	elif tag==2:
		x=125
		y=125
	elif tag==3:
		x=425
		y=125
	elif tag==4:
		x=425
		y=325
	else:
		x=425
		y=525
	return x,y				
	
if __name__ =="__main__":
	try:	
		rospy.init_node('bayes',anonymous=True)
		pub1 = rospy.Publisher('/marker_1',Marker, queue_size = 100)
		pub2 = rospy.Publisher('/marker_2',Marker, queue_size = 100)
		cube_list= Marker()
		cube_list.header.frame_id ="map"
		cube_list.header.stamp = rospy.Time.now()
		cube_list.ns="cubes"
		cube_list.type=cube_list.CUBE_LIST
		cube_list.action = cube_list.ADD
		cube_list.pose.orientation.w =1.0
		cube_list.scale.x = 0.2
		cube_list.scale.y = 0.2
		cube_list.scale.z = 0.2
		cube_list.color.g = 1.0
		cube_list.color.a = 1.0
		cube_list.id=0

		line_strip = Marker()
		line_strip.header.frame_id = "map"
		line_strip.header.stamp = rospy.Time.now()
		line_strip.ns="lines"
		line_strip.type=line_strip.LINE_STRIP
		line_strip.action = line_strip.ADD
		line_strip.pose.orientation.x = 0.0
		line_strip.pose.orientation.y = 0.0
		line_strip.pose.orientation.z = 0.0
		line_strip.pose.orientation.w = 1.0
		line_strip.scale.x = 0.05
		line_strip.scale.y = 0.05
		line_strip.color.b = 1.0
		line_strip.color.a = 1.0
		line_strip.id=1
		
		rospack = rospkg.RosPack()
		bag = rosbag.Bag(rospack.get_path('lab4')+'/bag/grid.bag')
		state = np.zeros((35,35,36))
		state[11][27][20]=1
		count=0
		
		s=3
		ind=int(s/2)
		g1=np.zeros((s,s,s))
		g2=np.zeros((s,s,s))
		for i in range(0,s):
			for j in range(0,s):
				for k in range(0,s):
					g1[i][j][k]=sp.multivariate_normal.pdf([i,j,k],[ind,ind,ind],[[0.25,0,0],[0,0.25,0],[0,0,0.25]])
					g2[i][j][k]=sp.multivariate_normal.pdf([i,j,k],[ind,ind,ind],[[1.25,0,0],[0,1.25,0],[0,0,0.6]])
					#g1[i][j][k]=sp.multivariate_normal.pdf([i,j,k],[ind,ind,ind],[[1,0,0],[0,1,0],[0,0,1]])
					#g2[i][j][k]=sp.multivariate_normal.pdf([i,j,k],[ind,ind,ind],[[1,0,0],[0,1,0],[0,0,1]])


		for topic, msg, t in bag.read_messages(topics=['Movements', 'Observations']):
			if msg.timeTag%2!=0:
				quaternion = (msg.rotation1.x, msg.rotation1.y, msg.rotation1.z, msg.rotation1.w)
				euler = tf.transformations.euler_from_quaternion(quaternion)
				theta1=euler[2]
				theta1=math.degrees(theta1)
				quaternion = (msg.rotation2.x, msg.rotation2.y, msg.rotation2.z, msg.rotation2.w)
				euler = tf.transformations.euler_from_quaternion(quaternion)
				theta2=euler[2]
				theta2=math.degrees(theta2) 
				state_temp = np.zeros((35,35,36))

				#Shifting probabilities
				for i in range(35):
			    		for j in range(35):
						for k in range(36):
							if state[i][j][k]>0:
								d=getAbsTheta(k)+theta1
								dx=msg.translation*100*math.cos(math.radians(d))
								dy=msg.translation*100*math.sin(math.radians(d))
								x=dx+getAbsLoc(i)
								y=dy+getAbsLoc(j)
								theta=getAbsTheta(k)+theta1+theta2
								if theta<0:
									theta=theta+360
								if theta>360:
									theta=theta-360
								if theta==360:
									theta=0
					
								x=getGridLoc(x)
								y=getGridLoc(y)
								theta=getGridTheta(theta)
								state_temp[x][y][theta]=state_temp[x][y][theta]+state[i][j][k]

				#Apply gaussian
				state_g = np.zeros((35,35,36))
				for i in range(0+ind,35-ind):
			    		for j in range(0+ind,35-ind):
						for k in range(0+ind,36-ind):
							if state_temp[i][j][k]>0:
								val=state_temp[i][j][k]
								inter=np.multiply(val,g1)
								for p,l in zip(range(i-ind,i+ind+1),range(s)):
									for q,m in zip(range(j-ind,j+ind+1),range(s)):
										for r,n in zip(range(k-ind,k+ind+1),range(s)):
											state_g[p][q][r]=state_g[p][q][r]+inter[l][m][n]
					

				max_ind=np.unravel_index(state_g.argmax(), state_g.shape)
		
			else:
				quaternion = (msg.bearing.x, msg.bearing.y, msg.bearing.z, msg.bearing.w)
				euler = tf.transformations.euler_from_quaternion(quaternion)
				b_angle=euler[2]

				max_g=np.unravel_index(state_g.argmax(), state_g.shape)
				r_angle=getAbsTheta(max_g[2])
				r_angle=math.radians(r_angle)
				tag_x, tag_y=getTagPos(msg.tagNum)
				dx=msg.range*100*math.cos(b_angle+r_angle)
				dy=msg.range*100*math.sin(b_angle+r_angle)
				x=tag_x-dx
				y=tag_y-dy
				x=int(getGridLoc(x))
				y=int(getGridLoc(y))
	
				theta_obs=getGridTheta(math.degrees(r_angle))
				state_obs=np.zeros((35,35,36))
				state_obs[x][y][theta_obs]=1
	
				#Applying gaussian
				state_obs_g = np.zeros((35,35,36))
				for i in range(0+ind,35-ind):
			    		for j in range(0+ind,35-ind):
						for k in range(0+ind,36-ind):
							if state_obs[i][j][k]>0:
								val=state_obs[i][j][k]
								inter=np.multiply(val,g2)
								for p,l in zip(range(i-ind,i+ind+1),range(s)):
									for q,m in zip(range(j-ind,j+ind+1),range(s)):
										for r,n in zip(range(k-ind,k+ind+1),range(s)):
											state_obs_g[p][q][r]=state_obs_g[p][q][r]+inter[l][m][n]
	

				state=np.add(state_obs_g,state_g)
				state=np.divide(state,2.)

				max_indices=np.unravel_index(state.argmax(), state.shape)
				x=max_indices[0]
				y=max_indices[1]
				z=max_indices[2]
				
				state_new = np.zeros((35,35,36))
				for i in range(x-ind,x+ind+1):
			    		for j in range(y-ind,y+ind+1):
						for k in range(z-ind,z+ind+1):
							state_new[i][j][k]=state[i][j][k]
					
				state_new=np.divide(state_new,np.sum(state_new))							
				state=state_new
				

				#rviz
				
				pt1=Point()
				pt1.x=1.25
				pt1.y=5.25
				cube_list.points.append(pt1)
				pt2=Point()
				pt2.x=1.25
				pt2.y=3.25
				cube_list.points.append(pt2)
				pt3=Point()
				pt3.x=1.25
				pt3.y=1.25
				cube_list.points.append(pt3)
				pt4=Point()
				pt4.x=4.25
				pt4.y=1.25
				cube_list.points.append(pt4)
				pt5=Point()
				pt5.x=4.25
				pt5.y=3.25
				cube_list.points.append(pt5)
				pt6=Point()
				pt6.x=4.25
				pt6.y=5.25
				cube_list.points.append(pt6)
				pub1.publish(cube_list)
				pt=Point()
				
				pt.x=0.01*getAbsLoc(max_indices[0])
				pt.y=0.01*getAbsLoc(max_indices[1])
				line_strip.points.append(pt)
				pub2.publish(line_strip)
		bag.close()
		
	except rospy.ROSInterruptException:
        	pass



