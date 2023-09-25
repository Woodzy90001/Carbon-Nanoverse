import sys
import time
import numpy as np
from scipy.spatial import KDTree


'''

This file takes an XYZ file and prepares it for use with Carbon Nanoverse.
Despite being designed for carbon, this will work for any atom type. 

Usage:
    python3 inputToCarbonNanoverse.py Input-file-name optional-args
    
args:
    --C       - Clean data. If not specified set to false
    --MC x    - Set a chunk size of x. If not specified 10 is used


Command line args:
    Input file name - The file to be processed. Must be an XYZ of format
                      specified on github:
                          https://github.com/Woodzy90001/Carbon-Nanoverse
                          
    Max chunk size  - The max chunk size allowed by the program.
    
    Clean data      - A flag to remove any bonds which are unrealistically 
                      long and any atoms with less than 2 connections for 
                      Carbon, and no connections for any other atom type.
'''


# Normal coorinates to chunk coorinates
def CoordsToChunkCoords(coords, chunkSize):
    return (coords / chunkSize).astype(int) # Casting to int rounds down

def ChunkCordsToIndex(chunkCoords, numChunks):
    return (chunkCoords[0] + 
            chunkCoords[1]*numChunks + 
            chunkCoords[2]*(numChunks**2))

def ParseBoxDimensions(splitData, unityCoords = False):
    retArray = np.array([float(splitData[0]), float(splitData[1]),
                     float(splitData[2])])
    if (unityCoords):
        retArray = np.array([float(splitData[0]), float(splitData[2]),
                         float(splitData[1])])
    return retArray

# Find equivelant chunk within original box
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

# Find chunk coord for position
def ChunkifyPoints(chunkNum, boxDimensions, points):
    infoArray = np.zeros((chunkNum, chunkNum, chunkNum))
    chunkSize = boxDimensions / chunkNum
    
    for point in points:
        indices = CoordsToChunkCoords(point, chunkSize)
        xInd, yInd, zInd = CentreChunkPoints(indices, chunkNum)
        
        infoArray[xInd][yInd][zInd] += 1
        
    return infoArray.reshape(-1)







def CleanAtomAndBondData(atoms, connections, atomType, bondDict):
    
    oldAtomIDs = np.array([])
    newAtomIDs = np.asarray(range(len(atoms)))
    
    while not (np.array_equal(oldAtomIDs, newAtomIDs)):
        
        # Remove atoms with too few connections
        oldAtomIDs = newAtomIDs
        newAtomIDs = np.full((len(atoms)), False)
        for i in oldAtomIDs:
            if ("C_" in atomType[i]):
                if (connections[i] >= 2):
                    newAtomIDs[i] = True
            else:
                if (connections[i] >= 0):
                    newAtomIDs[i] = True
                    
                    
        # Remove bonds attached to non existent atoms
        bondsToDelete = []
        for currBond in bondDict.values():
            atomDoesntExist = False
            atom1ID, atom2ID = currBond.GetIDInt()
            if not newAtomIDs[atom1ID]: atomDoesntExist = True
            if not newAtomIDs[atom2ID]: atomDoesntExist = True
            if (atomDoesntExist):
                bondsToDelete.append([currBond.GetID(), atom1ID, atom2ID])

                    
        for bond in bondsToDelete:
            del bondDict[bond[0]]
            connections[bond[1]] -= 1
            connections[bond[2]] -= 1
            
        
        tempList = []
        for ind in range(len(newAtomIDs)):
            if newAtomIDs[ind]: tempList.append(ind)
        newAtomIDs = np.asarray(tempList)
        
    # Delete any mismatch bonds
    
    # Calculate implied connections
    calculatedConnections = np.full((len(atoms)), 0)
    for currBond in bondDict.values():
        atom1ID, atom2ID = currBond.GetIDInt()
        calculatedConnections[atom1ID] += 1
        calculatedConnections[atom2ID] += 1
    
    bondsToDelete = []
    for i in newAtomIDs:
        # If there is a discrepancy between the calculated and impled 
        # connections, delete bond if both atoms it's connected to has a
        # discrepancy, or if all connected atoms have the right amount,
        # delete the longest bond. If there are less bonds then there are
        # meant to be, then ....... uhhh. 
        if (connections[i] != calculatedConnections[i]):
            strI = str(i)
            connectedBonds = [value for key, value in bondDict.items() if strI in key]
            removeBond = None
            
            for currBond in connectedBonds:
                atom1ID, atom2ID = currBond.GetIDInt()
                if ((connections[atom1ID] != calculatedConnections[atom1ID]) and (connections[atom2ID] != calculatedConnections[atom2ID])): 
                    removeBond = currBond
            
            maxDist = 0
            if removeBond is None:
                for currBond in connectedBonds:  
                    currDist = currBond.GetDistance()
                    if (currDist > maxDist):
                        maxDist = currDist
                        removeBond = currBond
            
            bondDataToAdd = removeBond.GetID()
            if bondDataToAdd not in bondsToDelete: bondsToDelete.append(bondDataToAdd)
    
    for bond in bondsToDelete:
        del bondDict[bond]
        
    return connections, newAtomIDs, bondDict
    
    




# Take an input file smaller than the max chunk size and copy the contents
# until it is larger than the max chunk size
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
                              float(splitData[2]), 
                              float(splitData[3]), 
                              int(float(splitData[4]))]))
                inputFileAtomTypes.append(splitData[0].capitalize())
                
                # This makes sure the coordinates are a float
            except Exception as e:
                print("Error on line {}".format(lineCount))
                print(e)
                sys.exit()
                
        # Save bounding box size
        elif (len(splitData) == 3):
            boxDimensions = ParseBoxDimensions(splitData)

    # Centre at 0
    minValues = np.amin(inputFileAtoms, axis=0)[:3]
    for i in range(len(inputFileAtoms)):
        inputFileAtoms[i][:3] -= minValues
        
    # Duplicate atoms to build box
    repeatTimes = [int(-(-maxChunkSize // chunkSize)) for chunkSize in boxDimensions]
    repeatTimes = np.asarray(repeatTimes)
    atomTypesRepeated = []
    atomsRepeated = []
    finalBoxSize = repeatTimes*boxDimensions
    for i in range(len(inputFileAtoms)):
        for xOffset in range(repeatTimes[0]):
            for yOffset in range(repeatTimes[1]):
                for zOffset in range(repeatTimes[2]):
                    newAtom = np.copy(inputFileAtoms[i])
                    newAtom[:3] += np.array([xOffset, yOffset, zOffset]) * boxDimensions
                    
                    if not np.any(np.greater_equal(newAtom[:3], finalBoxSize)):
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
    
    newFileName = "{}_repeated_C_{}.xyz".format(inputFile[:-4], str(min(finalBoxSize)).replace(".", "-"))
    f = open(newFileName, 'w')
    f.write("{}\n{} {} {}\n".format(len(atomsRepeated), finalBoxSize[0], finalBoxSize[1], finalBoxSize[2]))
    for i in range(len(atomsRepeated)):
        f.write("{} {} {} {} {}\n".format(atomTypesRepeated[i], atomsRepeated[i][0],
                                        atomsRepeated[i][1], atomsRepeated[i][2], 
                                        atomsRepeated[i][3],))

    f.close()    
    
    del(inputFileAtoms)
    del(inputFileAtomTypes)
    del(atomTypesRepeated)
    del(atomsRepeated)
    del(finalBoxSize)
    
    return newFileName


# Finnd max chunk size which ensures less than 1000 points in it
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
    
    if debugInfo:
        print("Optimal chunks found. Chunk num - {}\nTime taken - {}".format(chunkNum-1, time.time() - startChunkFind))
    
    numChunks = chunkNum - 1
    
    return numChunks

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
                testAtomUpdated[testIndex] += boxDimensions[testIndex]
                if (GetDistance(referenceAtom, testAtomUpdated) < bondDistance):
                    bondDistance = GetDistance(referenceAtom, testAtomUpdated)
                else:
                    testAtomUpdated[testIndex] -= boxDimensions[testIndex]*2
                    if (GetDistance(referenceAtom, testAtomUpdated) < bondDistance):
                        bondDistance = GetDistance(referenceAtom, testAtomUpdated)
                    else:
                        testAtomUpdated[testIndex] += boxDimensions[testIndex]
                    
            testAtom = testAtomUpdated
            
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
                if (min(midpoint) >= 0) and not np.any(np.greater(midpoint, boxDimensions)):
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
    
    def GetIDInt(self):
        return self.atom1ID, self.atom2ID
    
    def GetCoords(self):
        return [self.atom1Coords, self.atom2Coords]
    
    def AddOffset(self, offset):
        self.atom1Coords += offset
        self.atom2Coords += offset
        
    def Centre(self, boxDimensions):
        self.atom2Coords = FindCosestAtomLocation(self.atom1Coords, self.atom2Coords, boxDimensions)

        midpoint = self.GetMidpoint()
        if (min(midpoint) < 0) or np.any(np.greater(midpoint, boxDimensions)):
            self.atom1Coords, self.atom2Coords = CentreMidpoint(
                self.atom1Coords, self.atom2Coords, boxDimensions)
        
    def GetDistance(self):
        return GetDistance(self.atom1Coords, self.atom2Coords)




'''
Usage:
    python3 inputToCarbonNanoverse.py Input-file-name optional-args
    
args:
    --C       - Clean data. If not specified set to false
    --M x    - Set a chunk size of x. If not specified 10 is used
'''
maxChunkSize = 10
cleanData = False
debug = False

# Process command arguments

if ((len(sys.argv)) == 1) or (sys.argv[1][-4:] != ".xyz"):
    print("Please provide an XYZ filename. \n\nUsage:\n\tpython3 CNV-Unity.py Input-file-name optional-args\n\nargs:\n\t--C       - Clean data. If not specified set to false\n\t--MC x    - Set a chunk size of x. If not specified 10 is used")
    sys.exit()
if (len(sys.argv)) > 1:
    inputFile = sys.argv[1]
if (len(sys.argv)) > 2:
    for i in range(2, len(sys.argv)):
        if (sys.argv[i] == "--C"): cleanData = True
        if (sys.argv[i] == "--M") and (i < (len(sys.argv) - 1)): 
            try:
                maxChunkSize = float(sys.argv[i + 1])
            except:
                print("Error, max chunk size can't be parsed into a float. \n\nUsing 10 instead. ")

# Formatting output file name
cleanDataString = "N"
if cleanData: cleanDataString = "Y"
fileName = "{}-{}{}_NA.cnv".format(inputFile[:-4], str(maxChunkSize).replace(".", "-"), cleanDataString)

'''

~~~~~~~~~~~~~~~~~~~~~~~ File reading ~~~~~~~~~~~~~~~~~~~~~~~

'''

print("Reading file ...")
globalStart = time.time()
fileReadStart = time.time()





try:
    file = open(inputFile, "r")
    data = file.readlines()
    file.close()
except FileNotFoundError:
    print("Error, No such file or directory: {}".format(inputFile))
    sys.exit()
    

# Verify file is large enough
boxSizeFound = False
totalLineCount = len(data)
currLine = 0
while (not boxSizeFound and (currLine < totalLineCount)):
    line = data[currLine]
    splitData = SplitLine(line)

    if (len(splitData) == 3):
        boxDimensions = ParseBoxDimensions(splitData)
        boxSizeFound = True
    currLine += 1

if (min(boxDimensions) < maxChunkSize): 
    del(data)
    
    inputFile = RepeatInputFile(inputFile, maxChunkSize)
    try:
        file = open(inputFile, "r")
        data = file.readlines()
        file.close() 
    except FileNotFoundError:
        print("Error, No such file or directory: {}".format(inputFile))
        sys.exit()

del(boxDimensions)
validElements = ['h', 'he', 'li', 'be', 'b', 'c', 'n', 'o', 'f', 'ne', 'na', 'mg', 'al', 'si', 'p', 's', 'cl', 'ar', 'k', 'ca', 'sc', 'ti', 'v', 'cr', 'mn', 'fe', 'co', 'ni', 'cu', 'zn', 'ga', 'ge', 'as', 'se', 'br', 'kr', 'rb', 'sr', 'y', 'zr', 'nb', 'mo', 'tc', 'ru', 'rh', 'pd', 'ag', 'cd', 'in', 'sn', 'sb', 'te', 'i', 'xe', 'cs', 'ba', 'la', 'ce', 'pr', 'nd', 'pm', 'sm', 'eu', 'gd', 'tb', 'dy', 'ho', 'er', 'tm', 'yb', 'lu', 'hf', 'ta', 'w', 're', 'os', 'ir', 'pt', 'au', 'hg', 'tl', 'pb', 'bi', 'po', 'at', 'rn', 'fr', 'ra', 'ac', 'th', 'pa', 'u', 'np', 'pu', 'am', 'cm', 'bk', 'cf', 'es', 'fm', 'md', 'no', 'lr', 'rf', 'db', 'sg', 'bh', 'hs', 'mt', 'ds', 'rg', 'cn', 'nh', 'fl', 'mc', 'lv', 'ts', 'og']
atomTypes = []

atoms = []
atomType = []
connections = []
lineCount = 0



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
        boxDimensions = ParseBoxDimensions(splitData, unityCoords=True)

# Zero all atoms
minValues = np.amin(atoms, axis=0)
for i in range(len(atoms)):
    atoms[i] -= minValues
    
    
# Verify all points are within the bounds
mins = np.amin(atoms, axis=0)
maxs = np.amax(atoms, axis=0)
if (np.any(np.greater(maxs, boxDimensions))) or (np.any(np.less(mins, np.zeros(3)))):
    print("Error: Actual box dimensions are larger than expected dimensions. \n       Please fix input file.\n\nExpected dimension - {}\n\nActual dimensions : \n  x - {}\n  y - {}\n  z - {}".format(boxDimensions, maxs[0] - mins[0], maxs[1] - mins[1], maxs[2] - mins[2]))
    sys.exit()

maxChunkSizeChunkNum = max((np.ceil((boxDimensions / maxChunkSize))).astype(int))
print("File read done, time taken             - {}\n".format(time.time() - fileReadStart))










'''
Write start of outfile
'''
print("Creating output file ...")
fileCreateStart = time.time()
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
print("Output file created, time taken        - {}\n".format(time.time() - fileCreateStart))










'''

~~~~~~~~~~~~~~~~~~~~~~~ Bond calculation ~~~~~~~~~~~~~~~~~~~~~~~

'''

bondCount = 0
bondSum = 0

# Bond Dict for easy duplicate filtering
bondDict = {}
print("Finding bonds ...")

bondStart = time.time()
tree = KDTree(atoms, boxsize=boxDimensions)
distsRet, indexRet = tree.query(atoms, k=max(connections) + 1)

print("Bonds found, saving bonds ...")

chunkID = 0
atomNum = 0
for atom in atoms:
    newAtomObject = np.copy(atom)
    
    # Take the relevant indexes from the returned index list
    index = indexRet[atomNum][1:connections[atomNum]+1]
    
    # Take the relevant distances from the returned points list
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
        
        
        

if debug:  print("Bond length average - {}".format(bondSum/bondCount))
print("Bonds found, time taken                - {}\n".format(time.time() - 
                                                           bondStart))














'''

~~~~~~~~~~~~~~~~~~~~~~~ Atom and Bond chunking ~~~~~~~~~~~~~~~~~~~~~~~

'''



print("Cleaning atoms and finding chunk sizes ...")
atomClean = time.time()
# Calculate the chunk to ensure the maximum atom per chunk without exceding 
# 1000

atomsHolder = {} 

for at in atomTypes:
    atomsHolder[at] = []    

if (cleanData):
    connections, validAtomIDs, bondDict = CleanAtomAndBondData(atoms, connections, atomType, bondDict)
else:
    validAtomIDs = range(len(atoms))

for i in validAtomIDs:
    if ("C_" in atomType[i]):
        # The connectons may have been updated so update them
        currConnections = connections[i]
        if (currConnections == 2): atomTypeStr = "C_sp"
        elif (currConnections == 1): atomTypeStr = "C"
        else:
            atomTypeStr = "C_sp{}".format(currConnections - 1)
        atomType[i] = atomTypeStr
        
        if (connections[i] >= 2) or not cleanData:
            atomsHolder[atomType[i]].append(np.array([connections[i], atoms[i]], dtype=object))
    else:
        if (connections[i] >= 0) or not cleanData:
            atomsHolder[atomType[i]].append(np.array([connections[i], atoms[i]], dtype=object))

for atomTypeIndex in atomsHolder:
    points = np.take(atomsHolder[atomTypeIndex], [i*2 + 1 for i in range(len(atomsHolder[atomTypeIndex]))])

    numChunks = FindOptimalChunks(points, boxDimensions)
    
    if np.any(np.greater((boxDimensions / numChunks), maxChunkSize)):
        numChunks = maxChunkSizeChunkNum
    
    atomsHolder[atomTypeIndex].append(numChunks)
    
# atomsHolder now is [p1, p2, ..., pn, chunk size]
print("Atom cleaning done, time taken         - {}\n".format(time.time() - atomClean))








print("Assigning atoms a chunk ...")
atomAssignStart = time.time()

# Atom and Bond list for easy chunk based access

for atomName in atomsHolder:
    numChunks = atomsHolder[atomName][-1]
    currAtomsAndConnections = atomsHolder[atomName][:-1]
    chunkSizes = boxDimensions / numChunks
    currConnections = np.take(currAtomsAndConnections, [i*2 for i in range(len(currAtomsAndConnections))])
    currAtoms = np.take(currAtomsAndConnections, [i*2 + 1 for i in range(len(currAtomsAndConnections))])

    atomList = [[] for _ in range(numChunks**3)]
    atomIndex = 0
    for currAtom in currAtoms:
        try:
            chunkIndex = ChunkCordsToIndex(CoordsToChunkCoords(currAtom, chunkSizes), 
                                                               numChunks)
            '''
            [x, y, z, type, ID, conn]
            '''
            
            newAtom = np.array([float(currAtom[0]), 
                          float(currAtom[1]), 
                          float(currAtom[2]),
                          atomName,#atomType[atomIndex],
                          atomIndex,
                          currConnections[atomIndex]])
            atomList[chunkIndex].append(newAtom)
            
            
            atomIndex += 1

        except IndexError:
            print("Error: Actual box size larger than declared size.")
            sys.exit()


    try:
        f = open(fileName, "a")
        
        f.write("{}-{}-{}-{}-{}{{\n".format(atomName, str(numChunks), str(chunkSizes[0]), str(chunkSizes[1]), str(chunkSizes[2])))
        
        for chunk in atomList:
            for currAtom in chunk:
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
bondAssignStart = time.time()

bondMidpoints = []
for currBond in bondDict.values():
    midpoint = currBond.GetMidpoint()
    for i in range(len(midpoint)):
        if (midpoint[i] < 0):
            midpoint[i] = 0
        elif (midpoint[i] >= boxDimensions[i]):
            if (midpoint[i] - boxDimensions[i]) > 1:
                print("Error, bond too far out of bounds:\n\tBound - {}\n\tBond Loc - {}".format(boxDimensions[i], midpoint[i]))
                sys.exit()
            midpoint[i] = boxDimensions[i]-0.00001
    bondMidpoints.append(midpoint)

numChunks = FindOptimalChunks(bondMidpoints, boxDimensions)

if np.any(np.greater((boxDimensions / numChunks), maxChunkSize)):
    numChunks = maxChunkSizeChunkNum

chunkSizes = boxDimensions / numChunks
bondList = [[] for _ in range(numChunks**3)]

for currBond in bondDict.values():
    bondMidpoint = currBond.GetMidpoint()
    bondChunk = CoordsToChunkCoords(bondMidpoint, chunkSizes)
    chunkIndex = ChunkCordsToIndex(bondChunk, numChunks)
    bondList[chunkIndex].append(currBond)
    
try:
    f = open(fileName, "a")
    
    f.write("Bonds-{}-{}-{}-{}{{\n".format(str(numChunks), str(chunkSizes[0]), str(chunkSizes[1]), str(chunkSizes[2])))
    
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
