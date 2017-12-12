# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 11:29:26 2017

@author: yuyu-

The input of this file are the extracted ASTs (serialized) outputed by CodeSensor.
The output of this file are the textual sequence of ASTs.

"""

import os
import time

Project_Name = "Asterisk"

FILE_PATH = "D:\\" + Project_Name + "\\Vulnerable_Files\\Preprocessed\\"
Processed_FILE = "D:\\" + Project_Name + "\\Vulnerable_Files\\Processed\\"

def DepthFirstExtractASTs(file_to_process, file_name):
    
    lines = []
    subLines = ''
    
    f = open(file_to_process)
    try:
        original_lines = f.readlines()
        lines.append(file_name) # The first element is the file name.
        for line in original_lines:
            if not line.isspace(): # Remove the empty line.
                line = line.strip('\n')
                str_lines = line.split('\t')   
                #print (str_lines)
                if str_lines[0] != "water": # Remove lines starting with water.
                    #print (str_lines)
                    if str_lines[0] == "func":
                        # Add the return type of the function
                        subElement = str_lines[4].split() # Dealing with "static int" or "static void" or ...
                        if len(subElement) == 1:
                            lines.append(str_lines[4])
                        if subElement.count("*") == 0: # The element does not contain pointer type. If it contains pointer like (int *), it will be divided to 'int' and '*'.
                            if len(subElement) == 2:
                                lines.append(subElement[0])
                                lines.append(subElement[1]) 
                            if len(subElement) == 3:
                                lines.append(subElement[0])
                                lines.append(subElement[1])    
                                lines.append(subElement[2])
                        else:
                            lines.append(str_lines[4])
                        #lines.append(str_lines[5]) # Add the name of the function
                        lines.append("func_name") # Add the name of the function
                    if str_lines[0] == "params":
                        lines.append("params")                    
                    if str_lines[0] == "param":
                        subParamElement = str_lines[4].split() # Addd the possible type of the parameter
                        if len(subParamElement) == 1:
                            lines.append("param")
                            lines.append(str_lines[4]) # Add the parameter type
                        if subParamElement.count("*") == 0:
                            if len(subParamElement) == 2:
                                lines.append("param")
                                lines.append(subParamElement[0])
                                lines.append(subParamElement[1]) 
                            if len(subParamElement) == 3:
                                lines.append("param")
                                lines.append(subParamElement[0])
                                lines.append(subParamElement[1])    
                                lines.append(subParamElement[2])
                        else:
                            lines.append("param")
                            lines.append(str_lines[4]) # Add the parameter type                           
                    if str_lines[0] == "stmnts":
                        lines.append("stmnts")                    
                    if str_lines[0] == "decl":
                        subDeclElement = str_lines[4].split() # Addd the possible type of the declared veriable
                        #print (len(subDeclElement))
                        if len(subDeclElement) == 1:
                            lines.append("decl")
                            lines.append(str_lines[4]) # Add the type of the declared variable
                        if subDeclElement.count("*") == 0:
                            if len(subDeclElement) == 2:
                                lines.append("decl")
                                lines.append(subDeclElement[0])
                                lines.append(subDeclElement[1]) 
                            if len(subDeclElement) == 3:
                                lines.append("decl")
                                lines.append(subDeclElement[0])
                                lines.append(subDeclElement[1])    
                                lines.append(subDeclElement[2])
                        else:
                            lines.append("decl")
                            lines.append(str_lines[4]) # Add the type of the declared variable
                    if str_lines[0] == "op":
                        lines.append(str_lines[4])
                    if str_lines[0] == "call":
                        lines.append("call")
                        lines.append(str_lines[4])
                    if str_lines[0] == "arg":
                        lines.append("arg")
                    if str_lines[0] == "if":
                        lines.append("if")
                    if str_lines[0] == "cond":
                        lines.append("cond")
                    if str_lines[0] == "else":
                        lines.append("else")
                    if str_lines[0] == "stmts":
                        lines.append("stmts")
                    if str_lines[0] == "for":
                        lines.append("for") 	
                    if str_lines[0] == "forinit":
                        lines.append("forinit")
                    if str_lines[0] == "while":
                        lines.append("while")
                    if str_lines[0] == "return":
                        lines.append("return")
                    if str_lines[0] == "continue":
                        lines.append("continue")
                    if str_lines[0] == "break":
                        lines.append("break")
                    if str_lines[0] == "goto":
                        lines.append("goto")
                    if str_lines[0] == "forexpr":
                        lines.append("forexpr")
                    if str_lines[0] == "sizeof":
                        lines.append("sizeof")
                    if str_lines[0] == "do":
                        lines.append("do")   
                    if str_lines[0] == "switch":
                        lines.append("switch")   
                    if str_lines[0] == "typedef":
                        lines.append("typedef")
                    if str_lines[0] == "default":
                        lines.append("default")
                    if str_lines[0] == "register":
                        lines.append("register")
                    if str_lines[0] == "enum":
                        lines.append("enum")
                    if str_lines[0] == "union":
                        lines.append("union")
                    
                                          
        subLines = ','.join(lines)
        subLines = subLines + "," + "\n"
    finally:
        f.close()
        return subLines
 
    
big_line = []
total_processed = 0

for fpathe,dirs,fs in os.walk(FILE_PATH):
  for f in fs:
    if (os.path.splitext(f)[1]=='.txt'): # Get the .c files only
        file_path = os.path.join(fpathe,f) # f is the .c file, which will be processed by CodeSensor
        big_line.append(DepthFirstExtractASTs(FILE_PATH + f, f))
        #time.sleep(0.001)
        total_processed = total_processed + 1

print ("Totally, there are " + str(total_processed) + " files.")
strResult = ''.join(big_line)
f1 = open(Processed_FILE + "_proASTResult_DFT1.txt", "w")
f1.write(strResult)
f1.close()