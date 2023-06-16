#!/usr/bin/env python3

from __future__ import print_function # import the necessary modules
from beginner_tutorials.srv import *
from std_msgs.msg import String

import rospy
import math

increment = 0 # define increment start number

def inc(): # create a function that increments by 1 whenever it is called
    global increment
    increment += 1
    return increment

def timer_timer(data): # function that recieves time from node 3 and prints it
    rospy.loginfo("[Server 2 to Server 1]: %s" % (data.data))

def callback(req): # main calculator function

    try:
        m = str(req.first)

        m = m.replace("^", "**") #replace the users inputs with the operations needed to calculate
        m = m.replace("cbrt", "**(1/3)")
        m = m.replace("sqrt", "math.sqrt")
        m = m.replace("!", "math.factorial") 
        m = m.replace("e", "math.exp")
        m = m.replace("log2", "math.log2")
        m = m.replace("log10", "math.log10")

        ans = str(eval(m)) # evaluate the inputted equation

        i = inc() 

        rospy.loginfo("[Server 1]: Recieved %s. Calculated answer: %s" % (req.first, ans)) # print equation and answer server side

        operations = str("%s : Operation Number %s." %(ans, i)) # string that contains the answer and operation number

        text_file = open("Answers.txt", "a") # open or create a text file called answers
        text_file.write(f'{req.first} = {ans}\n') # save the equation and answer to the text file
        text_file.close() # close the text file

        resp = AddTwoIntsResponse()
        resp.sum = operations # send back the variable operations as a string
        return resp
    
    except: # if the equation cannot be evaluated becuase it is invalid
        invalidequation = "Please enter a Valid Equation!"
        ans = str(invalidequation)  
        
        rospy.loginfo("[Server 1]: Recieved %s, %s" % (req.first, ans)) # print what the user inputted and state that it is an invalid equation

        text_file = open("Answers.txt", "a") # open or create a text file called Answers
        text_file.write(f'{ans}: {req.first} is not valid!\n') # say that the input is invalid and show the user input, saved to the text file
        text_file.close() # close the text file

        resp = AddTwoIntsResponse()
        resp.sum = ans 
        return resp # send response back to client   

if __name__ == "__main__":
    rospy.init_node('adder') # create a node called adder
    rospy.Subscriber("chatter", String, timer_timer) # connect to publisher called chatter
    rospy.Service('calc', AddTwoInts, callback) # service called calc and run the function callback

    rospy.loginfo("Listening...") # print listening upon startup
    rospy.spin() # continue until told to close