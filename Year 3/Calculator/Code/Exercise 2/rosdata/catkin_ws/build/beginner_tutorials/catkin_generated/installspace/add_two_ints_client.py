#!/usr/bin/env python3

from __future__ import print_function # import needed modules
from beginner_tutorials.srv import *
from datetime import datetime, timedelta
from std_msgs.msg import String

import linecache
import sys
import rospy
import time

invalidequation = "Please enter a Valid Equation!"

now = datetime.now() # get current date and time
startup = now.strftime("%d/%m/%Y %H:%M:%S") # convert format of date and time

text_file = open("AllAnswers.txt", "a") # open or create a text file called AllAnswers
text_file.write(f'{startup}\n') # save the current date and time to the text file

howtouse = open("How_To_Use.txt", "r")
for x in howtouse:
    print(x)

def closing(): # function that prints the word closing
    rospy.loginfo("Closing...")

if __name__ == "__main__":
    rospy.init_node('caller') # create node called caller

    calc_client = rospy.ServiceProxy('calc', AddTwoInts) # call the service called calc
    counter_client = rospy.ServiceProxy('counter', AddTwoInts) # call the service called counter

    start = time.time() # 0s

    r = rospy.Rate(1)
    while not rospy.is_shutdown(): # while loop that loops while ros isnt shutdown
        a = input("Enter the Equation: ") # user inputs and equation

        if(a == "History" or a == "history"): # if the user inputs history then print the contents of the file answers to the client
            checkhistory = open("Answers.txt", "r")
            for x in checkhistory:
                rospy.loginfo(x)
            checkhistory.close()
            text_file = open("AllAnswers.txt", "a")
            text_file.write(f'The command {a} was carried out! \n') # save the command carried out to the text file
            text_file.close()            
            continue

        if(a == "Clear" or a == "clear"): # if the user inputs clear then clear/delete the contents of the file answers
            f = open("Answers.txt", "r+")
            f.truncate(0)
            rospy.loginfo("Log Cleared! \n")
            f.close()
            text_file = open("AllAnswers.txt", "a")
            text_file.write(f'The command {a} was carried out! \n') # save the command carried out to the text file
            text_file.close()  
            continue

        if(a == "Elements" or a == "elements"): # if the user inputs elements then print the number of elements in the file answers to the client
            file = open("Answers.txt", "r")
            line_count = 0
            for line in file:
                if line != "\n":
                    line_count += 1
            file.close()
            rospy.loginfo("Total number of elements: %s \n" % (line_count))
            text_file = open("AllAnswers.txt", "a")
            text_file.write(f'The command {a} was carried out! \n') # save the command carried out to the text file
            text_file.close()  
            continue    

        if(a == "Recall" or a == "recall"): # if the user inputs recall then read the contents of line chosen in file answers recalculate and display to the client
            recall_line = int(input("Which line would you like to recalculate?: "))
            line = linecache.getline("Answers.txt", recall_line)
            linecache.clearcache()
            rospy.loginfo("Recalculated: %s" % (line))
            text_file = open("AllAnswers.txt", "a")
            text_file.write(f'The command {a} was carried out! \n') # save the command carried out to the text file
            text_file.close()  
            continue
   
        if(a == "Time" or a == "time"): # if the user inputs time then print the server runtime and current date and time to the client
            end = time.time()
            run_time = (end-start)
            timeon = timedelta(seconds =run_time)
            now = datetime.now()
            datentime = now.strftime("%d/%m/%Y %H:%M:%S")
            rospy.loginfo("The server has been running for: %s. It is currently %s. \n" % (timeon, datentime))
            text_file = open("AllAnswers.txt", "a")
            text_file.write(f'The command {a} was carried out! \n') # save the command carried out to the text file
            text_file.close()  
            continue

        if(a == "Close" or a == "close"): # if the user inputs close then close the client
            text_file = open("AllAnswers.txt", "a")
            text_file.write(f'The command {a} was carried out! \n') # save the command carried out to the text file
            text_file.close() 
            rospy.on_shutdown(closing)
            break

        req = AddTwoIntsRequest()
        req.first = str(a) # send the user input

        qer = AddTwoIntsRequest()
        qer.tenth = float(start) # send the start time

        resp = calc_client(req) # recieve the answer
        reso = counter_client(qer) # recieve the runtime

        if(resp.sum == invalidequation):
            rospy.loginfo("[Client]: %s is not valid, %s" % (a, resp.sum)) # display the equation and answer on the client
            rospy.loginfo("[Client]: System has been running for: [%s] \n" % (reso.sum)) # display the system runtime on the client
            text_file = open("AllAnswers.txt", "a")
            text_file.write(f'{resp.sum} {a} is not valid!\n')
            text_file.close() # open file called AllAnswers, save the equation and the fact that it was not valid to the file
        else:
            rospy.loginfo("[Client]: %s = %s" % (a, resp.sum)) # display the equation and answer on the client
            rospy.loginfo("[Client]: System has been running for: [%s] \n" % (reso.sum)) # display the system runtime on the client
            text_file = open("AllAnswers.txt", "a")
            text_file.write(f'{a}={resp.sum}\n')
            text_file.close() # open file called AllAnswers, save the equation and answer to the file

        r.sleep() # wait for next input