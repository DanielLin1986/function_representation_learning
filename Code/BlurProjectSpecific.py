# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 20:36:39 2017

@author: yuyu-

"""

Project_name = ""

File_path = "D:\\" + Project_name + "\\Non-vulnerable_Files\\Processed\\"

def mapNodesToNumber(element):
    if element.count(" *") == 1:
        return "81" # pointer
    if element.count(" * *") == 1:
        return "82" # 2-rank pointer
    if element.count(" *") == 0 and element.count(" * *") == 0:
        if element == "":
            return "0"
        if element == "func_name":
            return "1"
        if element == "static":
            return "2"
        if element == "int":
            return "3"
        if element == "decl":
            return "4"
        if element == "if":
            return "5"
        if element == "else":
            return "6"
        if element == "while":
            return "7"
        if element == "do":
            return "8"
        if element == "for":
            return "9"
        if element == "forinit":
            return "10"
        if element == "cond":
            return "11"
        if element == "forexpr":
            return "12"
        if element == "return":
            return "13"
        if element == "call":
            return "14"
        if element == "continue":
            return "15"
        if element == "break":
            return "16"
        if element == "params":
            return "17"
        if element == "param":
            return "19"
        if element == "stmnts":
            return "20"
        if element == "arg":
            return "21"
        if element == "u_long":
            return "22"
        if element == "u_char":
            return "23"
        if element == "uint8":
            return "24"
        if element == "uint16":
            return "25"
        if element == "uint32":
            return "26"
        if element == "char":
            return "27"
        if element == "signed":
            return "28"
        if element == "unsigned":
            return "29"
        if element == "short":
            return "30"
        if element == "long":
            return "31"
        if element == "float":
            return "32"
        if element == "double":
            return "33"
        if element == "const":
            return "34"
        if element == "goto":
            return "35"
        if element == "sizeof":
            return "36"
        if element == "typedef":
            return "37"
        if element == "enum":
            return "38"
        if element == "register":
            return "39"
        if element == "union":
            return "40"
        if element == "switch":
            return "41"
        if element == "default":
            return "42"
        if element == "stmts":
            return "43"
        if element == "stmts":
            return "44"
        # map the operator
        if element == "+":
            return "45"
        if element == "-":
            return "46"
        if element == "*":
            return "47"
        if element == "/":
            return "48"
        if element == "%":
            return "49"
        if element == "++":
            return "50"
        if element == "--":
            return "51"
        if element == "=":
            return "52"
        if element == "==":
            return "53"
        if element == "!=":
            return "54"
        if element == ">":
            return "55"
        if element == "<":
            return "56"
        if element == ">=":
            return "57"
        if element == "<=":
            return "58"
        if element == "&&":
            return "59"
        if element == "||":
            return "60"
        if element == "&":
            return "61"
        if element == "|":
            return "62"
        if element == "^":
            return "63"
        if element == "~":
            return "64"
        if element == "<<":
            return "65"
        if element == ">>":
            return "66"
        if element == "+=":
            return "67"
        if element == "-=":
            return "68"
        if element == "*=":
            return "69"
        if element == "/=":
            return "70"
        if element == "%=":
            return "71"
        if element == "<<=":
            return "72"
        if element == ">>=":
            return "73"
        if element == "&=":
            return "74"
        if element == "^=":
            return "75"
        if element == "|=":
            return "76"
        if element == "?:":
            return "77"
        if element == "[":
            return "78"
        if element == "]":
            return "79"
        if element == ".":
            return "80"
        if element == "\n":
            return "84" # Specifying the ending of each function.
        else:
            return "83" # proj_specific type 
        
f = open(File_path + "_proASTResult_DFT.txt")

bigLines = []
newBigLines = ""

try:
    original_lines = f.readlines()
    for line in original_lines:
        if not line.isspace(): # Remove the empty line.
            subLine = line.split(',')
            print ()
            lines = []
            for index, item in enumerate(subLine):
                if index == 0:
                    lines.append(subLine[0]) # The first element is the function name.
                else:
                    lines.append(mapNodesToNumber(item))  
            newSubLine = ','.join(lines)
            #newSubLine = newSubLine + "\n"
        bigLines.append(newSubLine)
    newBigLines = '\n'.join(bigLines)
        
finally:
    f.close()
    
f1 = open(File_path + "_proASTResult_num_DFT.txt", "w")
f1.write(newBigLines)
f1.close()