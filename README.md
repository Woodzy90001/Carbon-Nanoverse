# Carbon Nanoverse Input File Parser

CNV-Unity converts XYZ files to input files for Carbon Nanoverse. The XYZ files must be cubic, ie have the same size x, y and z dimensions and be of the format:<br />
<br />
atom count<br />
xdim ydim zdim<br />
X xpos ypos zpos connections<br />
X xpos ypos zpos connections<br />
X xpos ypos zpos connections<br />
X xpos ypos zpos connections<br />
<br />
eg<br />
<br />
4<br />
10 10 10<br />
C 1   2   3 3<br />
C 3   2   8 3<br />
C 2.5 2   3 2<br />
C 8.9 4.8 9 3<br />


zeroXYZ converts an XYZ file so it is centred about 0 with the optional argument to find the bounding box size. 
