from Bio.PDB.PDBParser import PDBParser
import numpy as np
import cv2
#import csv
import random
from tqdm import tqdm
#from distinctipy import distinctipy
#from matplotlib import pyplot as plt
#import time

###### BGR amino acid encoding table
# Colors generated for max contrast with
# distinctipy.get_colors(20, [(0, 0, 0)])

AAcolors = {
"ALA":(255,255,255),
"ARG":(0,255,0),
"ASN":(255,0,255),
"ASP":(255,128,0),
"CYS":(0,128,255),
"GLN":(128,191,128),
"GLU":(171,6,77),
"GLY":(1,7,187),
"HIS":(223,129,239),
"ILE":(242,255,5),
"LEU":(3,252,211),
"LYS":(6,121,63),
"MET":(128,128,0),
"PHE":(118,86,146),
"PRO":(125,213,248),
"SER":(128,255,0),
"THR":(248,226,135),
"TRP":(114,10,252),
"TYR":(240,77,138),
"VAL":(255,0,0)
}

### Import PDB structure

parser = PDBParser(PERMISSIVE=1)

filename = "./AF-P42212-F1-model_v4.pdb"
structure_id = filename.split("-")[1]
structure = parser.get_structure(structure_id, filename)

# Extract atomic 3d coordinates array

for model in structure:
    for chain in model:
        points_3d = []
        colorlist = []
        for residue in chain:
            resn = residue.get_resname()
            for atom in residue:
                if atom.get_name() in ["N", "CA", "C"]:
                    atom_xyz = atom.get_coord()
                    points_3d.append(atom_xyz)
                    colorlist.append(resn)

points_3d = np.array(points_3d)
for i in range(len(colorlist)):
    colorlist[i] = AAcolors.get(colorlist[i])
#colorlist = np.array(colorlist, dtype = int)

###### Photography ######

# Define intrinsic camera parameters (currently set for pinhole)
focal_length = 600  # will determine if structure fits, ~zoom level
image_width = 1024
image_height = image_width
intrinsic_matrix = np.array([ 
	[focal_length, 0, image_width/2],  # camera x vector 
	[0, focal_length, image_height/2],  # camera y vector
	[0, 0, 1]  # camera z vector (aim @ PDB centroid?)
])

# Establishing stereo poses/wiggling
# Default rvec (L) is [0,0,0], rvecR [1,1,1] for now.
rvec_list = []
wiggle = 0.1
q = 1  # stereoscopic offset factor

for shutter in range(999):
    rvec_list.append([random.uniform(0-wiggle, wiggle),
                      random.uniform(0-wiggle, wiggle),
                      random.uniform(0-wiggle, wiggle)])

# Right stereoscopic view, rvec_list + q
rvec_listR = [[coord+q for coord in row] for row in rvec_list]

for shot in tqdm(range(len(rvec_list))):

# Define extrinsic camera parameters
    tvec = np.array([0, 0, 100], dtype=np.float32)  # x, y, zoom
    rvec = np.array(rvec_list[shot], dtype=np.float32)
    rvecR = np.array(rvec_listR[shot], dtype=np.float32) 

### Project 3D points onto 2D plane 
    points_2d, _ = cv2.projectPoints(points_3d, 
								rvec, tvec.reshape(-1, 1), 
								intrinsic_matrix, 
								None) 

    points_2dR, _ = cv2.projectPoints(points_3d, 
								rvecR, tvec.reshape(-1, 1), 
								intrinsic_matrix, 
								None)

### Plot 2D points as L/R images  
    imgL = np.zeros([image_height, image_width, 3], dtype=np.uint8)
    imgR = np.zeros([image_height, image_width, 3], dtype=np.uint8)
    ptSize = 2  # Size of amino acid points
    fill = -1  # -1 = filled, else stroke width in px
    k = 0
    for point in points_2d.astype(int):
        imgL = cv2.circle(imgL, tuple(point[0]), ptSize, colorlist[k], fill)
        k += 1
    k = 0
    for point in points_2dR.astype(int):
        imgR = cv2.circle(imgR, tuple(point[0]), ptSize, colorlist[k], fill)
        k += 1
    
    imname = structure_id + "_pose" + str(shot) + "_stereo"
    stereo = cv2.hconcat([imgL, imgR])
    cv2.imshow(imname, stereo)
    cv2.waitKey(8)
    cv2.destroyAllWindows()

