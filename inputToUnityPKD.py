import os
import sys
import time
import math
import datetime
import numpy as np
from operator import add # I cann replace this with numpy add for even more speed
from periodic_kdtree import PeriodicKDTree



        
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
    
    def addOffset(self, offset):
        self.atom1Coords = list(map(add, self.atom1Coords, offset))
        self.atom2Coords = list(map(add, self.atom2Coords, offset))

def CoordsToChunkCoords(coords, chunkSize):
    return [math.floor(coords[0] / chunkSize), 
            math.floor(coords[1] / chunkSize), 
            math.floor(coords[2] / chunkSize)]
    
def ChunkCordsToIndex(chunkCoords, numChunks):
    return (chunkCoords[0] + 
            chunkCoords[1]*numChunks + 
            chunkCoords[2]*(numChunks**2))
def CoordsToChunkCoordsBR(coords, chunkSize, numChunks):
    coordsTemp = CoordsToChunkCoords(coords, chunkSize)
    for i in range(len(coordsTemp)):
        if (coordsTemp[i] < 0):
            coordsTemp[i] = 0
        elif (coordsTemp[i] >= numChunks):
            coordsTemp[i] = numChunks-1
            
    return coordsTemp
    
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
inputFile = "1_5.xyz"
inputFile = "many_screws_4000K_0.24ns.xyz"
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
                          float(splitData[2])]))
            atomType.append(splitData[0])

            connections.append(int(float(splitData[4])))

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


offsetArray = np.array([minX*-1, minY*-1, minZ*-1])

for currAtom in atoms:
    currAtom = np.add(currAtom, offsetArray)


totalOutString += "File read done, time taken             - {}\n\n".format(time.time() - fileReadStart)
print("File read done, time taken             - {}\n".format(time.time() - fileReadStart))















'''

~~~~~~~~~~~~~~~~~~~~~~~ Bond calculation ~~~~~~~~~~~~~~~~~~~~~~~

'''

# Bond Dict for easy duplicate filtering
bondDict = {}

totalOutString += "Finding bonds ..."
print("Finding bonds ...")

bondStart = time.time()
searchDepth = np.ones((3))*5
tree = PeriodicKDTree(searchDepth, atoms, boxDimensions)
distsRet, indexRet, neighPointsRet = tree.query_outer(atoms, k=max(connections) + 1)

print("Bonds found, saving bonds ...")

chunkID = 0

atomNum = 0
for atom in atoms:
    
    # Clear terminal and print progress
    '''
    if (len(atoms) >= 100):
        if (atomNum % int(len(atoms)/100) == 0):
            print("\033[H\033[J")
            os.system('cls' if os.name == 'nt' else 'clear')
            print(totalOutString)
            print("Atom {} of {} - {:.2f}% Done".format(atomNum, len(atoms), 100*atomNum/len(atoms)))
    else:
        print("\033[H\033[J")
        os.system('cls' if os.name == 'nt' else 'clear')
        print(totalOutString)
        print("Atom {} of {} - {:.2f}% Done".format(atomNum, len(atoms), 100*chunkID/len(atoms)))
    '''
    
    
    index = indexRet[atomNum][1:connections[atomNum]+1]
    neighPoints = neighPointsRet[atomNum][1:connections[atomNum]+1]
    
    count = 0
    for point in neighPoints:
        newBond = Bond(atom, point, atomNum, index[count])
        
        if newBond.getID() not in bondDict:
                    bondDict[newBond.getID()] = newBond
                    
        count += 1
        
        
    atomNum += 1
        
        
        
        
        
            
'''
print("\033[H\033[J")
os.system('cls' if os.name == 'nt' else 'clear')
print(totalOutString)
print("Atom {} of {} - 100% Done".format(len(atoms), len(atoms)))
'''
print("Bonds found, time taken                - {}\n".format(time.time() - 
                                                           bondStart))














'''

~~~~~~~~~~~~~~~~~~~~~~~ Atom and Bond chunking ~~~~~~~~~~~~~~~~~~~~~~~

'''



# Decide the number of chunks to have a chunk size close to the ideal

numChunks = math.floor(boxDimensions / idealChunkSize)
chunkSize = boxDimensions / numChunks


print("Assigning atoms a chunk ...")

atomAssignStart = time.time()

# Atom and Bond list for easy chunk based access
atomList = [[] for _ in range(numChunks**3)]
bondList = [[] for _ in range(numChunks**3)]




atomIndex = 0
for currAtom in atoms:
    try:
        t = CoordsToChunkCoords(currAtom, chunkSize)
        chunkIndex = ChunkCordsToIndex(t, numChunks)
        #chunkIndex = ChunkCordsToIndex(CoordsToChunkCoords(currAtom, chunkSize), 
                                                           #numChunks)
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


print("Atom chunk assignment done, time taken - {}\n".format(time.time() - atomAssignStart))




print("Assigning bonds a chunk ...")

bondAssignStart = time.time()

for currBond in bondDict.values():
    bondMidpoint = currBond.getMidpoint()
    
    bondChunk = CoordsToChunkCoordsBR(bondMidpoint, chunkSize, numChunks)
    
    chunkIndex = ChunkCordsToIndex(bondChunk, numChunks)
    
    bondList[chunkIndex].append(currBond)

print("Bond chunk assignment done, time taken - {}\n".format(time.time() - bondAssignStart))


print("File write start...")
# Save atoms and boinds to final list

writeStart = time.time()





outText = ""
fileName = "formattedFile-{date:%Y-%m-%d_%H-%M-%S}.txt".format(date=datetime.datetime.now())

try:
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
        '''
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
        '''
        bondChunk = CoordsToChunkCoordsBR(currBondCoords, chunkSize, numChunks)
        
        currChunkID = ChunkCordsToIndex(bondChunk, numChunks)
        
        f.write("{} | {} | {} | {} | {} | {} | {}\n".format(coords[0][0],
                                                               coords[0][1],
                                                               coords[0][2],
                                                               coords[1][0],
                                                               coords[1][1],
                                                               coords[1][2],
                                                               currChunkID))
        
    
    f.close()
except Exception as e:
    f.close()
    print("Error writing file.")
    print(e)
    sys.exit()

print("File written, time taken               - {}\n".format(time.time() - 
                                                           writeStart))


print("Total time taken                       - {}".format(time.time() - 
                                                           globalStart))