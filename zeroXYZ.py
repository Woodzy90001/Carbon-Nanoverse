import sys
import numpy as np

'''

This file zeros all atoms for the CNV-Unity script. Also has optional paramater
to find the bounding box size.  

Usage:
    python3 neroXYZ.py Input-file-name --BS(optional)
    
Command line args:
    Input file name - The file to be processed. Must be an XYZ of format
                      specified on github:
                          https://github.com/Woodzy90001/Carbon-Nanoverse
                          
    --BS            - Find the bounding box size.
    
'''

findBoxSize = False
if ((len(sys.argv)) == 1) or (sys.argv[1][-4:] != ".xyz"):
    print("Please provide an XYZ filename. \n\nUsage:\n\tpython3 zeroXYZ.py Input-file-name --BS(optional)\n\n--BS\t- Find the box size. ")
    sys.exit()
if (len(sys.argv) > 1):
    inputFile = sys.argv[1]
if (len(sys.argv) == 3):
    findBoxSize = True
if (len(sys.argv) > 3):
    print("Error: too many arguments. \n\nUsage:\n\tpython3 zeroXYZ.py Input-file-name --BS(optional)\n\n--BS\t- Find the box size. ")
    sys.exit()
    
f = open(inputFile, 'r')
data = f.readlines()
f.close()

count = 0
boxSize = ""
auxData = []
atoms = []

for line in data:
    if (line != "\n"):
        if (count == 1):
            if (findBoxSize): boxSize = ""
            else: boxSize = line
        elif (count != 0):
            splitData = line.split(" ")
            if len(splitData) == 1: splitData = line.split("\t")
            splitData[-1] = splitData[-1].strip()
            splitData = list(filter(None, splitData))
            print(splitData)
            
            atoms.append(np.array([float(splitData[1]), float(splitData[2]), float(splitData[3])]))
            auxData.append([splitData[0]])
            for i in range(4, len(splitData)):
                auxData[-1].append(splitData[i])
    
    count += 1
    
minValues = np.amin(atoms, axis=0)
for i in range(len(atoms)):
    atoms[i] -= minValues
    
if (findBoxSize):
    maxs = np.amax(atoms, axis=0)
    boxSize = "{} {} {}".format(maxs[0], maxs[1], maxs[2])

f = open(inputFile[:-4] + "_zeroed.xyz", 'w')
f.write(str(len(atoms)) + "\n")
f.write(boxSize.strip() + "\n")
for i in range(len(atoms)):
    outStr = "{} {} {} {} ".format(auxData[i][0], atoms[i][0], atoms[i][1], atoms[i][2])
    for j in range(1, len(auxData[i])):
        outStr += str(auxData[i][j]) + " "
    f.write(outStr.strip() + "\n")
f.close()