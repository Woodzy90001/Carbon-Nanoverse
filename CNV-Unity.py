import sys
import time
import numpy as np
from scipy.spatial import KDTree


'''

This file takes an XYZ file and prepares it for use with Carbon Nanoverse.
Despite being designed for carbon, this will work for any atom type. 

Usage:
    python3 inputToCarbonNanoverse.py (Input File Name) (Max Chunk Size) (Clean Data)

Where the input file name, max chunk size and clean data are optional. 
If none are specified, the defaults will be used

Command line args:
    Input file name - The file to be processed. Must be an XYZ.
    Max chunk size  - The max chunk size allowed by the program
    Clean data      - A flag to remove any bonds which are unrealistically 
                      long and any atoms with less than 2 connections for 
                      Carbon, and no connections for any other atom type

The XYZ file must have the number of connections for each atom. 

'''


        
def CoordsToChunkCoords(coords, chunkSize):###########################
    return (coords / chunkSize).astype(int) # Casting to int rounds down
    
def ChunkCordsToIndex(chunkCoords, numChunks):
    return (chunkCoords[0] + 
            chunkCoords[1]*numChunks + 
            chunkCoords[2]*(numChunks**2))

def CentreChunkPoints(coords, chunkNum):
    for i in range(len(coords)):
        if (coords[i] >= chunkNum): coords[i] = chunkNum-1
        elif (coords[i] < 0): coords[i] = 0

    return coords

def SplitLine(line):
    splitData = line.split(" ")
    if len(splitData) == 1: splitData = line.split("\t")
    splitData[-1] = splitData[-1].strip()
    splitData = list(filter(None, splitData))
    return splitData

def ChunkifyPoints(chunkNum, boxDimensions, points):
    infoArray = np.zeros((chunkNum, chunkNum, chunkNum))
    chunkSize = boxDimensions / chunkNum
    
    for point in points:
        indices = CoordsToChunkCoords(point, chunkSize)
        xInd, yInd, zInd = CentreChunkPoints(indices, chunkNum)
        
        infoArray[xInd][yInd][zInd] += 1
        
    return infoArray.reshape(-1)
    
    





def RepeatInputFile(inputFile, maxChunkSize):
    
    f = open(inputFile, 'r')
    data = f.readlines()
    f.close()
    
    inputFileAtoms = []
    inputFileAtomTypes = []
    lineCount = 0
    for line in data:
        lineCount += 1
        
        # Strip and clean the input 
        splitData = SplitLine(line)
        
        if (len(splitData) == 5):
            
            '''
             Make atom object and add it to the atoms list. Ensure the types are
             correct on creation. 
            
            Exit the program if a wrong type is detected
            '''
            try:
                inputFileAtoms.append(np.array([float(splitData[1]), 
                              float(splitData[3]), 
                              float(splitData[2]), 
                              int(float(splitData[4]))]))
                inputFileAtomTypes.append(splitData[0].capitalize())
                
                # This makes sure the coordinates are a float
            except Exception as e:
                print("Error on line {}".format(lineCount))
                print(e)
                sys.exit()
                
        # Save bounding box size
        elif (len(splitData) == 3):
            boxDimensions = float(splitData[0]) # Assuming all side lengths are 
                                                # equal

    minValues = np.amin(inputFileAtoms, axis=0)[:3]
    for i in range(len(inputFileAtoms)):
        inputFileAtoms[i][:3] -= minValues
            
    repeatTimes = int(-(-maxChunkSize // boxDimensions)) # Round up without math import
    atomTypesRepeated = []
    atomsRepeated = []
    for i in range(len(inputFileAtoms)):
        for xOffset in range(repeatTimes):
            for yOffset in range(repeatTimes):
                for zOffset in range(repeatTimes):
                    newAtom = np.copy(inputFileAtoms[i])
                    newAtom[:3] += np.array([xOffset, yOffset, zOffset]) * boxDimensions
                    
                    if not (max(newAtom[:3]) >= repeatTimes*boxDimensions):
                        atomsRepeated.append(newAtom)
                        atomTypesRepeated.append(inputFileAtomTypes[i])
    
    # Remove any duplicate points
    allAtomData = [np.array([atomsRepeated[i][0], atomsRepeated[i][1], 
                               atomsRepeated[i][2], atomsRepeated[i][3], 
                               atomTypesRepeated[i]]) for i in range(len(atomsRepeated))]

    tuples = [tuple(row) for row in allAtomData]
    uniquePoints = np.unique(tuples, axis=0)
    
    # Tuples cast all values to strings. This casts them back to floats
    atomTypesRepeated = []
    atomsRepeated = []
    for point in uniquePoints:
        atomTypesRepeated.append(point[-1])
        atomsRepeated.append(np.array([float(point[0]), float(point[1]), float(point[2]), int(float(point[3]))]))
    
    newFileName = "{}_repeated_C_{}.xyz".format(inputFile[:-4], str(repeatTimes*boxDimensions).replace(".", "-"))
    f = open(newFileName, 'w')
    f.write("{}\n{} {} {}\n".format(len(atomsRepeated), repeatTimes*boxDimensions, repeatTimes*boxDimensions, repeatTimes*boxDimensions))
    for i in range(len(atomsRepeated)):
        f.write("{} {} {} {} {}\n".format(atomTypesRepeated[i], atomsRepeated[i][0],
                                        atomsRepeated[i][1], atomsRepeated[i][2], 
                                        atomsRepeated[i][3],))

    f.close()    
    
    del(inputFileAtoms)
    del(inputFileAtomTypes)
    del(atomTypesRepeated)
    del(atomsRepeated)
    
    return newFileName



def FindOptimalChunks(points, boxDimensions, sliceVal = 20, cutoff = 0.4, debugInfo=False):

    if debugInfo:
        print("Finding optimal chunks ...")
        startChunkFind = time.time()
    # Find num atoms for a given subdivision of chunks
    chunkNum = 1
    
    capacityArrays = []
    maxFound = False
    while not maxFound:
    
        
        infoArray = ChunkifyPoints(chunkNum, boxDimensions, points)
        
        if (max(infoArray) < 1000):
            maxFound = True
            
        capacityArrays.append(infoArray)
        chunkNum += 1
        
    capacityArrays.append(ChunkifyPoints(chunkNum, boxDimensions, points))
    
    if debugInfo:
        print("Optimal chunks found. Chunk num - {}\nTime taken - {}".format(chunkNum-1, time.time() - startChunkFind))
    
    maxAtoms = max(capacityArrays[chunkNum-2])
    minAtoms = min(capacityArrays[chunkNum-2])
    meanAtoms = np.mean(capacityArrays[chunkNum-2])
    stdAtoms = np.std(capacityArrays[chunkNum-2])
    
    numChunks = chunkNum - 1
    
    return maxAtoms, minAtoms, meanAtoms, stdAtoms, numChunks

def GetDistance(atom1, atom2):
    return np.linalg.norm(atom1-atom2)

def GetMidpoint(atom1, atom2):
    return (atom1 + atom2) / 2

# If there is a periodic bond, change the second atom location until the bond
# dist has been minimised and the proper geometry is obtained
def FindCosestAtomLocation(referenceAtom, testAtom, boxDimensions):
        bondDistance = GetDistance(referenceAtom, testAtom)
        if (bondDistance > 3):
            testAtomUpdated = testAtom
            
            for testIndex in range(3):
                testAtomUpdated[testIndex] += boxDimensions
                if (GetDistance(referenceAtom, testAtomUpdated) < bondDistance):
                    bondDistance = GetDistance(referenceAtom, testAtomUpdated)
                else:
                    testAtomUpdated[testIndex] -= boxDimensions*2
                    if (GetDistance(referenceAtom, testAtomUpdated) < bondDistance):
                        bondDistance = GetDistance(referenceAtom, testAtomUpdated)
                    else:
                        testAtomUpdated[testIndex] += boxDimensions
                    
            testAtom = testAtomUpdated
            
        #if (bondDistance > 3):
        #    print(bondDistance)
            
        return testAtom

# If the centrepoint of the bond is outside the bounding box then move the
# bond into the box
def CentreMidpoint(atom1, atom2, boxDimensions):

    for xOffset in range(-1,2,1):
        for yOffset in range(-1,2,1):
            for zOffset in range(-1,2,1):
                offsetVect = np.array([xOffset, yOffset, zOffset]) * boxDimensions
                testAtom1 = atom1 + offsetVect
                testAtom2 = atom2 + offsetVect
                
                midpoint = GetMidpoint(testAtom1, testAtom2)
                if (min(midpoint) >= 0) and (max(midpoint) <= boxDimensions):
                    break
            else: # I hate how this looks but it works
                continue
            break
        else:
            continue
        break
    
    return testAtom1, testAtom2
    
class Bond:
    
    def __init__(self, atom1Coords, atom2Coords, atom1ID, atom2ID):
        self.atom1Coords = atom1Coords
        self.atom2Coords = atom2Coords
        self.atom1ID = atom1ID
        self.atom2ID = atom2ID
        
    def GetMidpoint(self):
        return GetMidpoint(self.atom1Coords, self.atom2Coords)
        
    def GetID(self):
        if (self.atom1ID < self.atom2ID) : retID = "{}-{}".format(self.atom1ID,
                                                                  self.atom2ID)
        else : retID = "{}-{}".format(self.atom2ID, self.atom1ID)
        
        return retID
    
    def GetCoords(self):
        return [self.atom1Coords, self.atom2Coords]
    
    def AddOffset(self, offset):
        self.atom1Coords += offset
        self.atom2Coords += offset
        
    def Centre(self, boxDimensions):
        self.atom2Coords = FindCosestAtomLocation(self.atom1Coords, self.atom2Coords, boxDimensions)
        
        midpoint = self.GetMidpoint()
        if (min(midpoint) < 0) or (max(midpoint) > boxDimensions):
            self.atom1Coords, self.atom2Coords = CentreMidpoint(
                self.atom1Coords, self.atom2Coords, boxDimensions)
        
    def GetDistance(self):
        return GetDistance(self.atom1Coords, self.atom2Coords)





'''
Set a desired chunk size, the program will make the chunk size as close to
this as possible. 

Reccomended ~ 10
'''

maxChunkSize = 10
cleanData = False
#inputFile = "screw.xyz"
inputFile = "1_5.xyz"
#inputFile = "0_5.xyz"
inputFile = "diamondUnitCell.xyz"
inputFile = "cleanedLiC.xyz"

# Process command arguments
if (len(sys.argv)) > 1:
    inputFile = sys.argv[1]
if (len(sys.argv)) > 2:
    maxChunkSize = int(sys.argv[2])
if (len(sys.argv)) == 4:
    if (sys.argv[3].lower() == "true") or (sys.argv[3].lower() == "t") or (sys.argv[3].lower() == "y"):
        cleanData = True
    elif (sys.argv[3].lower() == "false") or (sys.argv[3].lower() == "f") or (sys.argv[3].lower() == "n"):
        cleanData = False



cleanDataString = "N"
if cleanData: cleanDataString = "Y"
fileName = "{}_{}{}.cnv".format(inputFile[:-4], maxChunkSize, cleanDataString)
print("Reading file ...")

'''

~~~~~~~~~~~~~~~~~~~~~~~ File reading ~~~~~~~~~~~~~~~~~~~~~~~

'''
globalStart = time.time()
fileReadStart = time.time()

file = open(inputFile, "r")
data = file.readlines()
file.close() 

# Verify file is large enough
boxSizeFound = False
totalLineCount = len(data)
currLine = 0
while (not boxSizeFound and (currLine < totalLineCount)):
    line = data[currLine]
    splitData = SplitLine(line)

    if (len(splitData) == 3):
        boxDimensions = float(splitData[0]) # Assuming all side lengths are 
                                            # equal
        boxSizeFound = True
    currLine += 1

if (boxDimensions < maxChunkSize): 
    del(data)
    
    inputFile = RepeatInputFile(inputFile, maxChunkSize)
    file = open(inputFile, "r")
    data = file.readlines()
    file.close() 

validElements = ['h', 'he', 'li', 'be', 'b', 'c', 'n', 'o', 'f', 'ne', 'na', 'mg', 'al', 'si', 'p', 's', 'cl', 'ar', 'k', 'ca', 'sc', 'ti', 'v', 'cr', 'mn', 'fe', 'co', 'ni', 'cu', 'zn', 'ga', 'ge', 'as', 'se', 'br', 'kr', 'rb', 'sr', 'y', 'zr', 'nb', 'mo', 'tc', 'ru', 'rh', 'pd', 'ag', 'cd', 'in', 'sn', 'sb', 'te', 'i', 'xe', 'cs', 'ba', 'la', 'ce', 'pr', 'nd', 'pm', 'sm', 'eu', 'gd', 'tb', 'dy', 'ho', 'er', 'tm', 'yb', 'lu', 'hf', 'ta', 'w', 're', 'os', 'ir', 'pt', 'au', 'hg', 'tl', 'pb', 'bi', 'po', 'at', 'rn', 'fr', 'ra', 'ac', 'th', 'pa', 'u', 'np', 'pu', 'am', 'cm', 'bk', 'cf', 'es', 'fm', 'md', 'no', 'lr', 'rf', 'db', 'sg', 'bh', 'hs', 'mt', 'ds', 'rg', 'cn', 'nh', 'fl', 'mc', 'lv', 'ts', 'og']
atomTypes = []

atoms = []
atomType = []
connections = []
lineCount = 0

minX = 100
minY = 100
minZ = 100

'''
[x, y, z]
type
ID
conn
'''

for line in data:
    lineCount += 1
    
    # Strip and clean the input 
    splitData = SplitLine(line)
    
    if (len(splitData) == 5):
        
        '''
         Make atom object and add it to the atoms list. Ensure the types are
         correct on creation. 
        
        Exit the program if a wrong type is detected
        '''
        try:
            atoms.append(np.array([float(splitData[1]), 
                          float(splitData[3]), 
                          float(splitData[2])]))
            
            connections.append(int(float(splitData[4])))
            
            atomTypeStr = splitData[0].capitalize()
            if (atomTypeStr == "C"):
                if (connections[-1] == 2): atomTypeStr = "C_sp"
                elif (connections[-1] == 1): atomTypeStr = "C"
                else:
                    atomTypeStr = "C_sp{}".format(connections[-1] - 1)
                    
            if atomTypeStr not in atomTypes:
                atomTypes.append(atomTypeStr)
                
            # A carbon atom in a material can't have only 1 connection so this
            # culls them if clean data is selected. 
            if (atomTypeStr == "C") and cleanData:
                atoms.pop()
                atomTypes.pop()
                connections.pop()
            else:
                atomType.append(atomTypeStr)
                
                

            # Save the min XYZ values to bring the box to 0
            if (float(splitData[1]) < minX):
                minX = float(splitData[1])
            if (float(splitData[3]) < minY):
                minY = float(splitData[3])
            if (float(splitData[2]) < minZ):
                minZ = float(splitData[2])
            
            
            
            # If the connections is a non integer, this check will fail and 
            # raise an error
            floatConnections = float(splitData[4])
            if (floatConnections != float(int(floatConnections))):
                print("Error on line {}".format(lineCount))
                print("Connections must be an integer")
                sys.exit()
               
                
               
            # If the atom type has a number in it this flag will be raised 
            # raise an error
            if splitData[0].lower() not in validElements:
                print("Error on line {}".format(lineCount))
                print("Atom type not in element list")
                sys.exit()
               
                
               
            # This makes sure the coordinates are a float
        except Exception as e:
            print("Error on line {}".format(lineCount))
            print(e)
            sys.exit()

    # Save number of atoms present
    elif (len(splitData) == 1):
        numAtoms = int(splitData[0])
        
    # Save bounding box size
    elif (len(splitData) == 3):
        boxDimensions = float(splitData[0]) # Assuming all side lengths are 
                                            # equal

print(atomTypes)

# Zero all atoms
offsetArray = np.array([minX*-1, minY*-1, minZ*-1])

zeroedAtoms = []
for currAtom in atoms:
    zeroedAtoms.append(np.add(currAtom, offsetArray))


atoms = zeroedAtoms

mins = [100,100,100]
maxs = [0,0,0]


# Verify data is within the specified chunk size.
for currAtom in atoms:
    if currAtom[0] > maxs[0]: maxs[0] = currAtom[0]
    if currAtom[0] < mins[0]: mins[0] = currAtom[0]
    if currAtom[1] > maxs[1]: maxs[1] = currAtom[1]
    if currAtom[1] < mins[1]: mins[1] = currAtom[1]
    if currAtom[2] > maxs[2]: maxs[2] = currAtom[2]
    if currAtom[2] < mins[2]: mins[2] = currAtom[2]

if (((maxs[0] - mins[0]) > boxDimensions) or ((maxs[1] - mins[1]) > boxDimensions) or ((maxs[2] - mins[2]) > boxDimensions)):
    print("Error: Actual box dimensions are larger than expected dimensions. \n       Please fix input file.\n\nExpected dimension - {}\n\nActual dimensions : \n  x - {}\n  y - {}\n  z - {}".format(boxDimensions, maxs[0] - mins[0], maxs[1] - mins[1], maxs[2] - mins[2]))
    sys.exit()

    
maxChunkSizeChunkNum = (np.ceil((boxDimensions / maxChunkSize))).astype(int)


print("File read done, time taken             - {}\n".format(time.time() - fileReadStart))











'''
Write start of outfile
'''

outText = ""
try:
    f = open(fileName, "w")
    
    f.write(fileName + "\n")
    
    atomTypesString = ""
    for currAtom in atomTypes:
        atomTypesString += "{},".format(currAtom.capitalize())
    
    f.write("Atoms present - {}\nAtoms\n".format(atomTypesString.strip(",")))
    
    f.close()
except Exception as e:
    print("Error writing file")
    print(e)
    sys.exit()











'''

~~~~~~~~~~~~~~~~~~~~~~~ Bond calculation ~~~~~~~~~~~~~~~~~~~~~~~

'''

bondCount = 0
bondSum = 0

# Bond Dict for easy duplicate filtering
bondDict = {}
print("Finding bonds ...")

bondStart = time.time()
tree = KDTree(atoms, boxsize=np.array([boxDimensions, boxDimensions, boxDimensions]))
distsRet, indexRet = tree.query(atoms, k=max(connections) + 1)

print("Bonds found, saving bonds ...")

chunkID = 0
atomNum = 0
for atom in atoms:
    newAtomObject = np.copy(atom)
    
    # Take the relevant indexes from the returned index list
    index = indexRet[atomNum][1:connections[atomNum]+1]
    
    # Take the relevant distannces from the returned points list
    dists = distsRet[atomNum][1:connections[atomNum]+1]
    
    # Gather the atom locations
    neighPoints = []
    for testIndex in index:
        testAtom = atoms[testIndex]
        neighPoints.append(np.copy(testAtom))
    
    if (len(dists) > 0):
        count = 0
        nearestNeighbour = dists[0]
        for point in neighPoints:
            
            newBond = Bond(newAtomObject, point, atomNum, index[count])
            newBond.Centre(boxDimensions)
            # Cull the bond if it's over 3A.
            if ( cleanData and ((dists[count] > nearestNeighbour*1.55) or (dists[count] > 3) or (newBond.GetDistance() > 3)) ):
                connections[atomNum] -= 1
            else:
                # Save bond if it doesn't exist
                if newBond.GetID() not in bondDict:
                    bondDict[newBond.GetID()] = newBond
                    bondSum += dists[count]
                    bondCount += 1
                        
            count += 1
        
        
    atomNum += 1
        
        
        
        
print("Bond average")
print(bondSum/bondCount)    
print("Bonds found, time taken                - {}\n".format(time.time() - 
                                                           bondStart))














'''

~~~~~~~~~~~~~~~~~~~~~~~ Atom and Bond chunking ~~~~~~~~~~~~~~~~~~~~~~~

'''



# Calculate the chunk to ensure the maximum atom per chunk without exceding 
# 1000

atomTypesAtomHolder = []
atomsIndexReference = {} # This is not a great way to do it but it'll work
                             # for now

count = 0
for at in atomTypes:
    atomTypesAtomHolder.append([at])
    atomsIndexReference[at] = count    
    count += 1

for i in range(len(atoms)):
    if ("C_" in atomType[i]):
        if (connections[i] >= 2) or not cleanData:
            atomTypesAtomHolder[atomsIndexReference[atomType[i]]].append(atoms[i])
    else:
        if (connections[i] > 0) or not cleanData:
            atomTypesAtomHolder[atomsIndexReference[atomType[i]]].append(atoms[i])

count = 0
for atomTypePoints in atomTypesAtomHolder:
    
    atomTypePoints = atomTypePoints[1:] # Remove the label point

    maxAtoms, minAtoms, meanAtoms, stdAtoms, numChunks = FindOptimalChunks(
    atomTypePoints, boxDimensions)
    
    if (boxDimensions / numChunks) > maxChunkSize:
        numChunks = maxChunkSizeChunkNum
    
    atomTypesAtomHolder[count].insert(1, numChunks)
    count += 1
    
    
    
# Atom types holder now is [Atom descriptor, chunk size, p1, p2, ...]

print("Assigning atoms a chunk ...")
atomAssignStart = time.time()

# Atom and Bond list for easy chunk based access

for atomTypeCurr in atomTypesAtomHolder:
    numChunks = atomTypeCurr[1]
    atomName = atomTypeCurr[0]
    currAtoms = atomTypeCurr[2:]
    chunkSize = boxDimensions / numChunks

    atomList = [[] for _ in range(numChunks**3)]
    atomIndex = 0
    for currAtom in currAtoms:
        try:
            chunkIndex = ChunkCordsToIndex(CoordsToChunkCoords(currAtom, chunkSize), 
                                                               numChunks)
            '''
            [x, y, z, type, ID, conn]
            '''
            
            newAtom = np.array([float(currAtom[0]), 
                          float(currAtom[1]), 
                          float(currAtom[2]),
                          atomType[atomIndex],
                          atomIndex,
                          connections[atomIndex]])
            atomList[chunkIndex].append(newAtom)
            
            
            atomIndex += 1
        except IndexError:
            print("Error: Actual box size larger than declared size.")
            sys.exit()


    try:
        f = open(fileName, "a")
        
        f.write("{}-{}-{}{{\n".format(atomName, str(numChunks), str(chunkSize)))
        
        for chunk in atomList:
            for currAtom in chunk:
                if (int(currAtom[5]) > 1):
                    f.write("{}|{}|{}|{}|{}~".format(currAtom[4],currAtom[0],currAtom[1],currAtom[2],currAtom[5]))
            f.write("\n") # Empty chunks have a blank line. Helps Unity know when not to load anything
        
        f.write("}\n")
        
        f.close()
    except Exception as e:
        print("Error writing file")
        print(e)
        sys.exit()

print("Atom chunk assignment done, time taken - {}\n".format(time.time() - atomAssignStart))






print("Assigning bonds a chunk ...")


bondMidpoints = []
for currBond in bondDict.values():
    midpoint = currBond.GetMidpoint()
    for i in range(len(midpoint)):
        if (midpoint[i] < 0):
            midpoint[i] = 0
        elif (midpoint[i] >= boxDimensions):
            midpoint[i] = boxDimensions-0.00001
    bondMidpoints.append(midpoint)

maxBonds, minBonds, meanBonds, stdBonds, numChunks = FindOptimalChunks(
bondMidpoints, boxDimensions)

if (boxDimensions / numChunks) > maxChunkSize:
    numChunks = maxChunkSizeChunkNum

chunkSize = boxDimensions / numChunks
bondList = [[] for _ in range(numChunks**3)]
bondAssignStart = time.time()

for currBond in bondDict.values():
    bondMidpoint = currBond.GetMidpoint()
    bondChunk = CoordsToChunkCoords(bondMidpoint, chunkSize)
    chunkIndex = ChunkCordsToIndex(bondChunk, numChunks)
    bondList[chunkIndex].append(currBond)
    
try:
    f = open(fileName, "a")
    
    f.write("Bonds-{}-{}{{\n".format(str(numChunks), str(chunkSize)))
    
    for chunk in bondList:
        for currBond in chunk:
            [atom1Coords, atom2Coords] = currBond.GetCoords()
            
            f.write("{}|{}|{}|{}|{}|{}~".format(atom1Coords[0],atom1Coords[1],atom1Coords[2],atom2Coords[0],atom2Coords[1],atom2Coords[2]))
        f.write("\n")
    
    f.write("}\n")
    
    f.close()
except Exception as e:
    print("Error writing file")
    print(e)
    sys.exit()

print("Bond chunk assignment done, time taken - {}\n".format(time.time() - bondAssignStart))
print("Total time taken                       - {}".format(time.time() - 
                                                           globalStart))