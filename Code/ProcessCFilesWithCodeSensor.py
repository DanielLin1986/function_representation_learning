# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 15:38:30 2017

This file achieve two functionalities:
    1) Grab all the .c files from a specified directory
    2) Process all the .c files with CodeSensor and output them to a specified directory.

"""

import os
import shutil
from subprocess import Popen, PIPE, STDOUT

"""
Path

"""

Software_package = "FFmpeg"

Working_directory = "D:\\"

#SOURCE_FILE = "\\Vulnerable_Files\\"

SOURCE_FILE = "\\Non-vulnerable_Files\\Non-vul_functions\\"

CodeSensor_OUTPUT_PATH = Working_directory + Software_package + "\\Non-vulnerable_Files\\Preprocessed\\"

CodeSensor_PATH = "D:\\CodeSensor.jar"

PATH = Working_directory + Software_package + SOURCE_FILE
#file_path_arr = []

Full_path = ""

for fpathe,dirs,fs in os.walk(PATH):
  for f in fs:
    if (os.path.splitext(f)[1]=='.c'): # Get the .c files only
        file_path = os.path.join(fpathe,f) # f is the .c file, which will be processed by CodeSensor
        
        # With each .c file open, CodeSensor will process the opened file and output all the processed files to a specified directory.
        Full_path = CodeSensor_OUTPUT_PATH + "_" + f + ".txt"
        
        with open(Full_path, "w+") as output_file:
            Popen(['C:\\ProgramData\\Oracle\\Java\\javapath\\java', '-jar', CodeSensor_PATH, file_path], stdout=output_file, stderr=STDOUT)        
            output_file.close()
            
        
        
            #print (lines)        
#            f1 = open(CodeSensor_OUTPUT_PATH + "_" + f + ".txt")           
#        try:
#            print ("start !")
#            lines = f1.readlines()
#            for line in lines:
#                if line != "":
#                    print (line)
#                else:
#                    print ("The line is empty!")
#        finally:
#            f1.close()           
