#!/usr/bin/env python3

from __future__ import print_function # import needed modules
from beginner_tutorials.srv import *
from datetime import datetime, timedelta
from std_msgs.msg import String

import rospy
import math
import time

def callback(qer): # function that takes in qer

        ans = float(qer.tenth) # ans is a float variable

        end = time.time() # end is the time that this server recieves the input since the client connection
        elapsed_time = (end-ans) # calculates the server run time in seconds

        timein = timedelta(seconds =elapsed_time) # convert the format of server run time 
        
        now = datetime.now() # get the current date and time
        dt = now.strftime("%d/%m/%Y %H:%M:%S") # convert the format of the date and time

        rospy.loginfo("[Server 2]: Run Time = %s @ %s" % (timein, dt)) # print the system run time and current date and time

        timings = str("%s @ %s" % (timein, dt)) # create variable called timings that contains the system run time and current date and time

        pub.publish(str(timings)) # publish timings

        reso = AddTwoIntsResponse()
        reso.sum = str(timings) 
        return reso # send timings back to client

if __name__ == "__main__":

    rospy.init_node('timer') # create node called timer
    pub = rospy.Publisher('chatter', String, queue_size=10) # create publisher and assign to variable
    rospy.Service('counter', AddTwoInts, callback) # service called counter

    rospy.loginfo("Listening...") # print listening upon startup
    rospy.spin() # continue until closed
