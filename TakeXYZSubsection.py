import sys
import numpy as np

def SplitLine(line):
    splitData = line.split(" ")
    if len(splitData) == 1: splitData = line.split("\t")
    splitData[-1] = splitData[-1].strip()
    splitData = list(filter(None, splitData))
    return splitData

def ParseArguments(arguments):
    
    inputFile = arguments[1]

    xs = np.array([float(arguments[2]), float(arguments[3])])
    ys = np.array([float(arguments[4]), float(arguments[5])])
    zs = np.array([float(arguments[6]), float(arguments[7])])
    
    bounds = np.array([xs, ys, zs], dtype=object)
    
    return inputFile, bounds

# Take subsection of XYZ

zeroAtoms = False
if (not (((len(sys.argv)) == 8) or (len(sys.argv)) == 9)) or (sys.argv[1][-4:] != ".xyz"):
    print("Please provide an XYZ filename. \n\nUsage:\n\tpython3 TakeXYZSubsection.py Input-file-name x-min x-max y-min y-max z-min z-max optional-args\n\nargs:\n\t--Z       - Zero atoms. If not specified set to false")
    sys.exit()
elif (len(sys.argv)) == 8:
    try:
        inputFile, bounds = ParseArguments(sys.argv)
    except:
        print("Error: Input can't be parsed to float")
elif (len(sys.argv)) == 9:
    try:
        inputFile, bounds = ParseArguments(sys.argv)
    except:
        print("Error: Input can't be parsed to float")
    if (sys.argv[8] == "--Z"): zeroAtoms = True


fileName = "{}-culled.xyz".format(inputFile[:-4])

f = open(inputFile, 'r')
data = f.readlines()
f.close()

atomsPos = []
atomsOther = []
for line in data:
    splitLine = SplitLine(line)
    if len(splitLine) > 3:
        atomsPos.append(np.array([float(splitLine[1]), float(splitLine[2]), float(splitLine[3])]))
        otherData = [splitLine[0]]
        for i in range(4, len(splitLine)):
            otherData.append(splitLine[i])
        
        atomsOther.append(np.asarray(otherData))
        
    elif len(splitLine) == 3:
        fileBounds = [splitLine[0], splitLine[1], splitLine[2]]


verifiedAtoms = []
verifiedAtomsPos = []
for i in range(len(atomsPos)):
    currAtom = atomsPos[i]
    
    if ((currAtom[0] >= bounds[0][0]) and (currAtom[0] <= bounds[0][1])) and ((currAtom[1] >= bounds[1][0]) and (currAtom[1] <= bounds[1][1])) and ((currAtom[2] >= bounds[2][0]) and (currAtom[2] <= bounds[2][1])):
        verifiedAtoms.append(np.array([currAtom, atomsOther[i]], dtype=object))
        verifiedAtomsPos.append(currAtom)

if (zeroAtoms):
    np.asarray(verifiedAtomsPos)
    minValues = np.amin(verifiedAtomsPos, axis=0)
    for i in range(len(verifiedAtoms)):
        verifiedAtoms[i][0] -= minValues
        #verifiedAtomsPos[i] -= minValues
    
    maxValues = np.amax(verifiedAtomsPos, axis=0)
    fileBounds = maxValues

f = open(fileName, 'w')
f.write("{}\n{} {} {}\n".format(len(verifiedAtoms), fileBounds[0], fileBounds[1], fileBounds[2]))

for passedAtom in verifiedAtoms:
    writeString = "{} {} {} {}".format(passedAtom[1][0], passedAtom[0][0], passedAtom[0][1], passedAtom[0][2])
    for i in range(1, len(passedAtom[1])):
        writeString += " " + passedAtom[1][i]
    f.write(writeString + "\n")
    
f.close()
