import os
import sys
import time
import math
import copy
import heapq
import datetime
import numpy as np
from scipy.spatial import KDTree



        
class Bond:
    
    def __init__(self, atom1Coords, atom2Coords, atom1ID, atom2ID):
        self.atom1Coords = atom1Coords
        self.atom2Coords = atom2Coords
        self.atom1ID = atom1ID
        self.atom2ID = atom2ID
        
    def getMidpoint(self):
        return [(self.atom1Coords[0] + self.atom2Coords[0]) / 2,
                (self.atom1Coords[1] + self.atom2Coords[1]) / 2,
                (self.atom1Coords[2] + self.atom2Coords[2]) / 2,]
        
    def getID(self):
        if (self.atom1ID < self.atom2ID) : retID = "{}-{}".format(self.atom1ID,
                                                                  self.atom2ID)
        else : retID = "{}-{}".format(self.atom2ID, self.atom1ID)
        
        return retID
    
    def getCoords(self):
        return [self.atom1Coords, self.atom2Coords]


def CoordsToChunkCoords(coords, chunkSize):
    return [math.floor(coords[0] / chunkSize), 
            math.floor(coords[1] / chunkSize), 
            math.floor(coords[2] / chunkSize)]
    
def ChunkCordsToIndex(chunkCoords, numChunks):
    return (chunkCoords[0] + 
            chunkCoords[1]*numChunks + 
            chunkCoords[2]*(numChunks**2))

def IndexToChunkCoords(index, numChunks):
    z = math.floor(index / (numChunks**2))
    xy = index % numChunks**2
    y = math.floor(xy / numChunks)
    x = xy % numChunks
    return [x, y, z]





'''
Set a desired chunk size, the program will make the chunk size as close to
this as possible. 

Reccomended ~ 10
'''
idealChunkSize = 10

inputFile = "0_5.xyz"
#inputFile = "1_5.xyz"
#inputFile = "many_screws_4000K_0.24ns.xyz"
#inputFile = "screw_3500K_0.71ns.xyz"



totalOutString = "Reading file ...\n"
print("Reading file ...")


'''

~~~~~~~~~~~~~~~~~~~~~~~ File reading ~~~~~~~~~~~~~~~~~~~~~~~

'''
globalStart = time.time()
fileReadStart = time.time()

file = open(inputFile, "r")
data = file.readlines()
file.close() 



validElements = ['h', 'he', 'li', 'be', 'b', 'c', 'n', 'o', 'f', 'ne', 'na', 'mg', 'al', 'si', 'p', 's', 'cl', 'ar', 'k', 'ca', 'sc', 'ti', 'v', 'cr', 'mn', 'fe', 'co', 'ni', 'cu', 'zn', 'ga', 'ge', 'as', 'se', 'br', 'kr', 'rb', 'sr', 'y', 'zr', 'nb', 'mo', 'tc', 'ru', 'rh', 'pd', 'ag', 'cd', 'in', 'sn', 'sb', 'te', 'i', 'xe', 'cs', 'ba', 'la', 'ce', 'pr', 'nd', 'pm', 'sm', 'eu', 'gd', 'tb', 'dy', 'ho', 'er', 'tm', 'yb', 'lu', 'hf', 'ta', 'w', 're', 'os', 'ir', 'pt', 'au', 'hg', 'tl', 'pb', 'bi', 'po', 'at', 'rn', 'fr', 'ra', 'ac', 'th', 'pa', 'u', 'np', 'pu', 'am', 'cm', 'bk', 'cf', 'es', 'fm', 'md', 'no', 'lr', 'rf', 'db', 'sg', 'bh', 'hs', 'mt', 'ds', 'rg', 'cn', 'nh', 'fl', 'mc', 'lv', 'ts', 'og']
atoms = []
atomTypes = []
lineCount = 0

minX = 100
minY = 100
minZ = 100

'''
Atom formatting
self.atomType = atomType           # String
self.x = x                         # Float
self.y = y                         # Float
self.z = z                         # Float
self.connections = connections     # Int
self.ID = None                  # Int

[x, y, z, type, ID, conn]

'''

for line in data:
    lineCount += 1
    
    # Strip and clean the input 
    splitData = line.split(" ")
    splitData[-1] = splitData[-1].strip()
    splitData = list(filter(None, splitData))
    
    if (len(splitData) == 5):
        
        '''
         Make atom object and add it to the atoms list. Ensure the types are
         correct on creation. 
        
        Exit the program if a wrong type is detected
        '''
        try:
            atoms.append(np.array([float(splitData[1]), 
                          float(splitData[3]), 
                          float(splitData[2]),
                          splitData[0],
                          None,
                          int(float(splitData[4]))]))

            if splitData[0].lower() not in atomTypes:
                atomTypes.append(splitData[0].lower())
            


            # Save the min XYZ values to bring the box to 0
            if (float(splitData[1]) < minX):
                minX = float(splitData[1])
            if (float(splitData[2]) < minY):
                minY = float(splitData[2])
            if (float(splitData[3]) < minZ):
                minZ = float(splitData[3])
            
            
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



# Zero all atoms

''' 
for i in range(len(atoms)):
    atoms[i][0] -= minX*-1
    atoms[i][1] -= minY*-1
    atoms[i][2] -= minZ*-1
'''
for currAtom in atoms:
    currAtom[0] -= minX*-1
    currAtom[1] -= minY*-1
    currAtom[2] -= minZ*-1


totalOutString += "File read done, time taken             - {}\n".format(time.time() - fileReadStart)
print("File read done, time taken             - {}".format(time.time() - fileReadStart))















'''

~~~~~~~~~~~~~~~~~~~~~~~ Atom chunking ~~~~~~~~~~~~~~~~~~~~~~~

'''



# Decide the number of chunks to have a chunk size close to the ideal

numChunks = math.floor(boxDimensions / idealChunkSize)
chunkSize = boxDimensions / numChunks


totalOutString += "Assigning atoms a chunk ...\n"
print("Assigning atoms a chunk ...")

atomAssignStart = time.time()

# Atom and Bond list for easy chunk based access
atomList = [[] for _ in range(numChunks**3)]
bondList = [[] for _ in range(numChunks**3)]
# Bond Dict for easy duplicate filtering
bondDict = {}



atomIndex = 0
for currAtom in atoms:
    try:
        chunkIndex = ChunkCordsToIndex(CoordsToChunkCoords([currAtom[0],currAtom[1],currAtom[2]], 
                                                           chunkSize), 
                                                           numChunks)
        
        
        currAtom[4] = atomIndex
        atomList[chunkIndex].append(currAtom)
        
        
        atomIndex += 1
        
        '''
        
        If the actual distance between XMin and XMax exceeds the box size given
        on line 2 this error will be thrown. 
        
        eg.
        
        Input file states box size of 40.0
        
        Min X = -0.257634
        Max X =  40.2847
        
        Actual box size = 40.542334
        
        This causes sizing errors. To get around this try increasing the box size
        in line 2. The length of the bonds around the boundaries however 
        may not be accurate but the program will run.
        
        '''
    except IndexError:
        print("Error: Actual box size larger than declared size.")
        sys.exit()
    
    
totalOutString += "Atom chunk assignment done, time taken - {}\n".format(time.time() - atomAssignStart)
print("Atom chunk assignment done, time taken - {}\n".format(time.time() - atomAssignStart))






'''
Atom formatting
self.atomType = atomType           # String
self.x = x                         # Float
self.y = y                         # Float
self.z = z                         # Float
self.connections = connections     # Int
self.ID = None                  # Int

[x, y, z, type, ID, conn]

'''





'''

~~~~~~~~~~~~~~~~~~~~~~~ Bond calculation ~~~~~~~~~~~~~~~~~~~~~~~

'''


totalOutString += "Finding bonds ..."
print("Finding bonds ...")

bondStart = time.time()


chunkID = 0
for chunk in atomList:
    
    # Show report every ~ 1%
    if (len(atomList) >= 100):
        if (chunkID % int(len(atomList)/100) == 0):
            print("\033[H\033[J")
            os.system('cls' if os.name == 'nt' else 'clear')
            print(totalOutString)
            print("Starting chunk {}/{} - {:.2f}% Done".format(chunkID, len(atomList), 100*chunkID/len(atomList)))
    else:
        print("\033[H\033[J")
        os.system('cls' if os.name == 'nt' else 'clear')
        print(totalOutString)
        print("Starting chunk {}/{} - {:.2f}% Done".format(chunkID, len(atomList), 100*chunkID/len(atomList)))
    
    '''
    
    Make tree for given chunk
    
    
    '''
    
    chunkCoords = IndexToChunkCoords(chunkID, numChunks)
    
    # Only do inner bits for now
    #if (chunkCoords[0] > 0 and chunkCoords[0] != numChunks-1) and (chunkCoords[1] > 0 and chunkCoords[1] != numChunks-1) and (chunkCoords[2] > 0 and chunkCoords[2] != numChunks-1):
    
    
    
    xs = []
    ys = []
    zs = []
    testAtomPositionToIndex = []
    
    for xIndex in range(chunkCoords[0]-1, chunkCoords[0]+2):
        for yIndex in range(chunkCoords[1]-1, chunkCoords[1]+2):
            for zIndex in range(chunkCoords[2]-1, chunkCoords[2]+2):
                
                xOffset = 0
                yOffset = 0
                zOffset = 0
                
                xIndexCurr = xIndex
                yIndexCurr = yIndex
                zIndexCurr = zIndex
                
                if (xIndex < 0):
                    xOffset = -1*boxDimensions
                    xIndexCurr += numChunks
                elif (xIndex == numChunks):
                    xOffset = 1*boxDimensions
                    xIndexCurr -= numChunks
                    
                if (yIndex < 0):
                    yOffset = -1*boxDimensions
                    yIndexCurr += numChunks
                elif (yIndex == numChunks):
                    yOffset = 1*boxDimensions
                    yIndexCurr -= numChunks
                    
                if (zIndex < 0):
                    zOffset = -1*boxDimensions
                    zIndexCurr += numChunks
                elif (zIndex == numChunks):
                    zOffset = 1*boxDimensions
                    zIndexCurr -= numChunks
                    
                
                currChunkIndex = ChunkCordsToIndex([xIndexCurr, yIndexCurr, zIndexCurr], numChunks)
                currChunkAtoms = atomList[currChunkIndex]
                
                
                for testAtom in currChunkAtoms:
                    xs.append(testAtom[0] + xOffset)
                    ys.append(testAtom[1] + yOffset)
                    zs.append(testAtom[2] + zOffset)
                    
                    testAtomPositionToIndex.append(testAtom[4])
    
    souroundingAtoms = np.c_[xs, ys, zs]
    tree = KDTree(souroundingAtoms)
    
    
    
    
    '''
    
    Find NN
    
    '''
    
    
    for currAtom in chunk:
        dists, index = tree.query([currAtom[:3]], k=currAtom[5] + 1)
        index = index[0] # Double wrapped var for some reason
        
        # Add Bonds
        currAtomID = currAtom[4]
        currAtomCoords = [currAtom[0], currAtom[1], currAtom[2]]
        
        connectionCount = 0
        for i in range(len(index)):
            
            neighbourAtom = souroundingAtoms[index[i]]
            neighbourCoords = [neighbourAtom[0],
                               neighbourAtom[1],
                               neighbourAtom[2]]
            newBond = Bond(currAtomCoords, neighbourCoords, 
                           currAtomID, testAtomPositionToIndex[index[i]])
            
            bondList[chunkID].append(newBond)
            
            connectionCount += 1
    
    
    chunkID += 1
    
for chunk in bondList:
    for bond in chunk:
        if bond.getID() not in bondDict:
            bondDict[bond.getID()] = bond


print("\033[H\033[J")
os.system('cls' if os.name == 'nt' else 'clear')
print(totalOutString)
print("Starting chunk {}/{} - 100% Done".format(len(atomList), len(atomList)))

print("Bonds found, time taken                - {}".format(time.time() - 
                                                           bondStart))












# Save atoms and boinds to final list

writeStart = time.time()





outText = ""
fileName = "formattedFile-{date:%Y-%m-%d_%H-%M-%S}.txt".format(date=datetime.datetime.now())

f = open(fileName, "w")

f.write(fileName + "\n")

atomTypesString = ""
for currAtom in atomTypes:
    atomTypesString += "{},".format(currAtom)

f.write("Atoms present - {}\n".format(atomTypesString.strip(",")))
f.write("\n\nNumber of chunks - {}\nChunk size - {}\n\nAtoms\nAtom ID | X Coord | Y Coord | Z Coord | Connections | Chunk ID\n".format(numChunks, chunkSize))

chunkID = 0
for chunk in atomList:
    for atom in chunk:
        f.write("{} | {} | {} | {} | {} | {}\n".format(atom[4], atom[0], 
                                                       atom[1], atom[2], 
                                                       atom[5], chunkID))
        
    chunkID += 1
    
f.write("\n\nBonds\nAtom 1-X | Atom 1-Y | Atom 1-Z | Atom 2-X | Atom 2-Y | Atom 2-Z | Chunk ID\n")
for bond in bondDict.values():
    coords = bond.getCoords()
    currBondCoords = bond.getMidpoint()
    midPointCoords = CoordsToChunkCoords(currBondCoords, chunkSize)
    
    if (midPointCoords[0] < 0):
        midPointCoords[0] += numChunks
    elif (midPointCoords[0] >= numChunks):
        midPointCoords[0] -= numChunks
        
    if (midPointCoords[1] < 0):
        midPointCoords[1] += numChunks
    elif (midPointCoords[1] >= numChunks):
        midPointCoords[1] -= numChunks
        
    if (midPointCoords[2] < 0):
        midPointCoords[2] += numChunks
    elif (midPointCoords[2] >= numChunks):
        midPointCoords[2] -= numChunks
    
    currChunkID = ChunkCordsToIndex(midPointCoords, numChunks)
    
    f.write("{} | {} | {} | {} | {} | {} | {}\n".format(coords[0][0],
                                                           coords[0][1],
                                                           coords[0][2],
                                                           coords[1][0],
                                                           coords[1][1],
                                                           coords[1][2],
                                                           currChunkID))
    

f.close()

print("File written, time taken               - {}".format(time.time() - 
                                                           writeStart))


print("Total time taken                       - {}".format(time.time() - 
                                                           globalStart))