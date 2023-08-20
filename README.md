# Carbon Nanoverse Input File Parser

This file converts XYZ files to input files for Carbon Nanoverse. The XYZ files must be cubic, ie have the same size x, y and z dimensions and be of the format:

atom count
xdim ydim zdim
X xpos ypos zpos connections
X xpos ypos zpos connections
X xpos ypos zpos connections
X xpos ypos zpos connections

eg

4
10 10 10
C 1   2   3 3
C 3   2   8 3
C 2.5 2   3 2
C 8.9 4.8 9 3
