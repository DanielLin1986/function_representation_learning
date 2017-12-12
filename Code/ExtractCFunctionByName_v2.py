# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 14:32:50 2017

@author: yuyu-

This file is buggy. 

"""

# The source_file_path is the .c file which contains the functions to be extracted.
# The func_name is the name of the function that is to be extracted   
# The destination_file_path is the directory where the extracted function will be stored.     
def extractCFunctionByName(source_file_path, func_name, destination_file_path):
    list_function = []
    
    total_lines = ""

    starting_line = ""

    brace_left = 0
    brace_right = 0
    
    f = open(source_file_path, encoding="latin1")
    try:
        lines = []
        origin_lines = f.readlines()
        
        for line in origin_lines:
            if not line.isspace(): # Remove empty lines.
                lines.append(line)
#            print (line)
        total_lines = len(lines)
        print ("The fine contains: " + str(total_lines) + " lines.")
        for index, line in enumerate(lines):
            # If there is a line which contains the name of the function;
            
            if func_name.strip() in line.strip():
                print (line)
                #print ("\r\n")
                if ";" in line: #This line is usually function invokation or function declaration.
                    continue;
                if index != 0 and index != total_lines -1: # The function is not the first line and the last line of the file.        
                    if "{" in lines[index]:
                        if(lines[index-1].strip() != ""):
                            #print (lines[index-1].strip())
                            print ("The " + func_name + " starts from line: " + str(index-1)) # Usually it is the starting line of the function 
                            #print (lines[index-1])
                            list_function.append(lines[index-1])
                        else: 
                            print ("The " + func_name + " starts from line: " + str(index))
                        #print (line)
                        list_function.append(line)
                        # Include the case like: "return_type function_name(){} 
                        if "{" in line: 
                            brace_left = brace_left + 1
                        if "}" in line:
                            brace_right = brace_right + 1
                        starting_line = index
                        
                        #print (brace_left)
                        #print (brace_right)
                        #print (range(starting_line))
                        #sequence_range = chain(range(starting_line), range(total_lines))
                        #print(sequence_range)
                        for i in range(starting_line + 1, total_lines):
                            #print (lines[i].strip())
                            if "{" in lines[i].strip():
                                brace_left = brace_left + 1
                                #print (brace_left)
                            if "}" in lines[i].strip():
                                brace_right = brace_right + 1
                                #print (brace_left)
                            if (brace_left > brace_right):
                                #print (lines[i])  
                                list_function.append(lines[i])
                                continue                        
                            if(brace_left == brace_right):
                                #print (lines[i])
                                list_function.append(lines[i])
                                break
                            
                    if "{" in lines[index+1]: # Exclude function invokation and function declaration. Conventionally, "{" follows closely after function name.
                        if(lines[index-1].strip() != ""):
                            #print (lines[index-1].strip())
                            print ("The " + func_name + " starts from line: " + str(index-1)) # Usually it is the starting line of the function 
                            #print (lines[index-1])
                            list_function.append(lines[index-1])
                        else: 
                            print ("The " + func_name + " starts from line: " + str(index))
                        #print (line)
                        list_function.append(line)
                        # Include the case like: "return_type function_name(){} 
                        if "{" in line: 
                            brace_left = brace_left + 1
                        if "}" in line:
                            brace_right = brace_right + 1
                        starting_line = index
                        
                        #print (brace_left)
                        #print (brace_right)
                        #print (range(starting_line))
                        #sequence_range = chain(range(starting_line), range(total_lines))
                        #print(sequence_range)
                        for i in range(starting_line + 1, total_lines):
                            #print (lines[i].strip())
                            if "{" in lines[i].strip():
                                brace_left = brace_left + 1
                                #print (brace_left)
                            if "}" in lines[i].strip():
                                brace_right = brace_right + 1
                                #print (brace_left)
                            if (brace_left > brace_right):
                                #print (lines[i])  
                                list_function.append(lines[i])
                                continue                        
                            if(brace_left == brace_right):
                                #print (lines[i])
                                list_function.append(lines[i])
                                break
                    else:   
                        if ")" in lines[index+1] and "{" in lines[index+2]: 
                            if(lines[index-1].strip() != ""):
                                #print (lines[index-1].strip())
                                print ("The " + func_name + " starts from line: " + str(index-1)) # Usually it is the starting line of the function 
                                #print (lines[index-1])
                                list_function.append(lines[index-1])
                            else: 
                                print ("The " + func_name + " starts from line: " + str(index))
                            #print (line)
                            list_function.append(line)
                            list_function.append(lines[index+1])
                            # Include the case like: "return_type function_name(){} 
                            if "{" in line: 
                                brace_left = brace_left + 1
                            if "}" in line:
                                brace_right = brace_right + 1
                            starting_line = index
                            
                            #print (brace_left)
                            #print (brace_right)
                            #print (range(starting_line))
                            #sequence_range = chain(range(starting_line), range(total_lines))
                            #print(sequence_range)
                            for i in range(starting_line + 2, total_lines):
                                #print (lines[i].strip())
                                if "{" in lines[i].strip():
                                    brace_left = brace_left + 1
                                    #print (brace_left)
                                if "}" in lines[i].strip():
                                    brace_right = brace_right + 1
                                    #print (brace_left)
                                if (brace_left > brace_right):
                                    #print (lines[i])  
                                    list_function.append(lines[i])
                                    continue                        
                                if(brace_left == brace_right):
                                    #print (lines[i])
                                    list_function.append(lines[i])
                                    break
                        if index + 3 < len(lines): # Make sure the index will not exceed the length of the file.
                            if "," in lines[index+1] and ";" not in lines[index+1] and ")" in lines[index+2] and "{" in lines[index+3]:
                                if(lines[index-1].strip() != ""):
                                    #print (lines[index-1].strip())
                                    print ("The " + func_name + " starts from line: " + str(index-1)) # Usually it is the starting line of the function 
                                    #print (lines[index-1])
                                    list_function.append(lines[index-1])
                                else: 
                                    print ("The " + func_name + " starts from line: " + str(index))
                                #print (line)
                                list_function.append(line)
                                list_function.append(lines[index+1])
                                list_function.append(lines[index+2])
                                # Include the case like: "return_type function_name(){} 
                                if "{" in line: 
                                    brace_left = brace_left + 1
                                if "}" in line:
                                    brace_right = brace_right + 1
                                starting_line = index
                                
                                #print (brace_left)
                                #print (brace_right)
                                #print (range(starting_line))
                                #sequence_range = chain(range(starting_line), range(total_lines))
                                #print(sequence_range)
                                for i in range(starting_line + 3, total_lines):
                                    #print (lines[i].strip())
                                    if "{" in lines[i].strip():
                                        brace_left = brace_left + 1
                                        #print (brace_left)
                                    if "}" in lines[i].strip():
                                        brace_right = brace_right + 1
                                        #print (brace_left)
                                    if (brace_left > brace_right):
                                        #print (lines[i])  
                                        list_function.append(lines[i])
                                        continue                        
                                    if(brace_left == brace_right):
                                        #print (lines[i])
                                        list_function.append(lines[i])
                                        break
                        if index + 4 < len(lines):
                            if "," in lines[index+2] and ";" not in lines[index+2] and ")" in lines[index+3] and "{" in lines[index+4]:
                                if(lines[index-1].strip() != ""):
                                    #print (lines[index-1].strip())
                                    print ("The " + func_name + " starts from line: " + str(index-1)) # Usually it is the starting line of the function 
                                    #print (lines[index-1])
                                    list_function.append(lines[index-1])
                                else: 
                                    print ("The " + func_name + " starts from line: " + str(index))
                                #print (line)
                                list_function.append(line)
                                list_function.append(lines[index+1])
                                list_function.append(lines[index+2])
                                list_function.append(lines[index+3])
                                # Include the case like: "return_type function_name(){} 
                                if "{" in line: 
                                    brace_left = brace_left + 1
                                if "}" in line:
                                    brace_right = brace_right + 1
                                starting_line = index
                                
                                #print (brace_left)
                                #print (brace_right)
                                #print (range(starting_line))
                                #sequence_range = chain(range(starting_line), range(total_lines))
                                #print(sequence_range)
                                for i in range(starting_line + 4, total_lines):
                                    #print (lines[i].strip())
                                    if "{" in lines[i].strip():
                                        brace_left = brace_left + 1
                                        #print (brace_left)
                                    if "}" in lines[i].strip():
                                        brace_right = brace_right + 1
                                        #print (brace_left)
                                    if (brace_left > brace_right):
                                        #print (lines[i])  
                                        list_function.append(lines[i])
                                        continue                        
                                    if(brace_left == brace_right):
                                        #print (lines[i])
                                        list_function.append(lines[i])
                                        break
                        if index + 5 < len(lines):
                            if "," in lines[index+1] and "," in lines[index+2] and ";" not in lines[index+2] and ")" in lines[index+4] and "{" in lines[index+5]:
                                if(lines[index-1].strip() != ""):
                                    #print (lines[index-1].strip())
                                    print ("The " + func_name + " starts from line: " + str(index-1)) # Usually it is the starting line of the function 
                                    #print (lines[index-1])
                                    list_function.append(lines[index-1])
                                else: 
                                    print ("The " + func_name + " starts from line: " + str(index))
                                #print (line)
                                list_function.append(line)
                                list_function.append(lines[index+1])
                                list_function.append(lines[index+2])
                                list_function.append(lines[index+3])
                                list_function.append(lines[index+4])
                                #list_function.append(lines[index+5])
                                # Include the case like: "return_type function_name(){} 
                                if "{" in line: 
                                    brace_left = brace_left + 1
                                if "}" in line:
                                    brace_right = brace_right + 1
                                starting_line = index
                                
                                #print (brace_left)
                                #print (brace_right)
                                #print (range(starting_line))
                                #sequence_range = chain(range(starting_line), range(total_lines))
                                #print(sequence_range)
                                for i in range(starting_line + 5, total_lines):
                                    #print (lines[i].strip())
                                    if "{" in lines[i].strip():
                                        brace_left = brace_left + 1
                                        #print (brace_left)
                                    if "}" in lines[i].strip():
                                        brace_right = brace_right + 1
                                        #print (brace_left)
                                    if (brace_left > brace_right):
                                        #print (lines[i])  
                                        list_function.append(lines[i])
                                        continue                        
                                    if(brace_left == brace_right):
                                        #print (lines[i])
                                        list_function.append(lines[i])
                                        break
                        if index + 6 < len(lines):
                            if "," in lines[index+1] and "," in lines[index+2] and ";" not in lines[index+2] and ";" not in lines[index+3] and ")" in lines[index+5] and "{" in lines[index+6]:
                                if(lines[index-1].strip() != ""):
                                    #print (lines[index-1].strip())
                                    print ("The " + func_name + " starts from line: " + str(index-1)) # Usually it is the starting line of the function 
                                    #print (lines[index-1])
                                    list_function.append(lines[index-1])
                                else: 
                                    print ("The " + func_name + " starts from line: " + str(index))
                                #print (line)
                                list_function.append(line)
                                list_function.append(lines[index+1])
                                list_function.append(lines[index+2])
                                list_function.append(lines[index+3])
                                list_function.append(lines[index+4])
                                list_function.append(lines[index+5])
                                #list_function.append(lines[index+6])
                                # Include the case like: "return_type function_name(){} 
                                if "{" in line: 
                                    brace_left = brace_left + 1
                                if "}" in line:
                                    brace_right = brace_right + 1
                                starting_line = index
                                
                                #print (brace_left)
                                #print (brace_right)
                                #print (range(starting_line))
                                #sequence_range = chain(range(starting_line), range(total_lines))
                                #print(sequence_range)
                                for i in range(starting_line + 6, total_lines):
                                    #print (lines[i].strip())
                                    if "{" in lines[i].strip():
                                        brace_left = brace_left + 1
                                        #print (brace_left)
                                    if "}" in lines[i].strip():
                                        brace_right = brace_right + 1
                                        #print (brace_left)
                                    if (brace_left > brace_right):
                                        #print (lines[i])  
                                        list_function.append(lines[i])
                                        continue                        
                                    if(brace_left == brace_right):
                                        #print (lines[i])
                                        list_function.append(lines[i])
                                        break
                        if index + 7 < len(lines):
                            if "," in lines[index+1] and "," in lines[index+2] and ";" not in lines[index+1] and ";" not in lines[index+2] and ";" not in lines[index+3] and ")" in lines[index+6] and "{" in lines[index+7]:
                                if(lines[index-1].strip() != ""):
                                    #print (lines[index-1].strip())
                                    print ("The " + func_name + " starts from line: " + str(index-1)) # Usually it is the starting line of the function 
                                    #print (lines[index-1])
                                    list_function.append(lines[index-1])
                                else: 
                                    print ("The " + func_name + " starts from line: " + str(index))
                                #print (line)
                                list_function.append(line)
                                list_function.append(lines[index+1])
                                list_function.append(lines[index+2])
                                list_function.append(lines[index+3])
                                list_function.append(lines[index+4])
                                list_function.append(lines[index+5])
                                list_function.append(lines[index+6])
                                # Include the case like: "return_type function_name(){} 
                                if "{" in line: 
                                    brace_left = brace_left + 1
                                if "}" in line:
                                    brace_right = brace_right + 1
                                starting_line = index
                                
                                #print (brace_left)
                                #print (brace_right)
                                #print (range(starting_line))
                                #sequence_range = chain(range(starting_line), range(total_lines))
                                #print(sequence_range)
                                for i in range(starting_line + 7, total_lines):
                                    #print (lines[i].strip())
                                    if "{" in lines[i].strip():
                                        brace_left = brace_left + 1
                                        #print (brace_left)
                                    if "}" in lines[i].strip():
                                        brace_right = brace_right + 1
                                        #print (brace_left)
                                    if (brace_left > brace_right):
                                        #print (lines[i])  
                                        list_function.append(lines[i])
                                        continue                        
                                    if(brace_left == brace_right):
                                        #print (lines[i])
                                        list_function.append(lines[i])
                                        break
                """
                   The above second "if ")" in lines[index+1] and "{" in lines[index+2]:  " deals with the following case:
                   
                   void
                   func_name(int x, int y,
                             int z)
                   {
                      do something...
                   } 
                   
                   and the third if condition deals with:
                 
                   void
                   func_name(int x, int y,
                             int z, int a,
                             int b)
                   {}
                """
            #else:
                #print("match not found!")
    finally:
        f.close()  
    with open(destination_file_path, "w", encoding="latin1") as file:
        for item in list_function:
            file.write("%s" % item)
        file.close()

#working_dir = "D:\\Source_Code\\Pidgin\\pidgin-2.8.0\\"
#source_file = "libpurple\\plugins\\buddynote.c"
#destination_file = "test.c"
#func_name = "init_plugin"
#
#extractCFunctionByName(working_dir + source_file, func_name, "d:\\" + destination_file)