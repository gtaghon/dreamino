from Bio.PDB.PDBParser import PDBParser
import numpy as np
import pandas as pd
import cv2
import csv
import random
from tqdm import tqdm

### BGR amino acid encoding table
AAcolors = {
"ALA":(35,244,35),
"ARG":(46,233,46),
"ASN":(57,222,57),
"ASP":(68,211,68),
"CYS":(79,200,79),
"GLN":(90,189,90),
"GLU":(101,178,101),
"GLY":(112,167,112),
"HIS":(123,156,123),
"ILE":(134,145,134),
"LEU":(145,134,145),
"LYS":(156,123,156),
"MET":(167,112,167),
"PHE":(178,101,178),
"PRO":(189,90,189),
"SER":(200,79,200),
"THR":(211,68,211),
"TRP":(222,57,222),
"TYR":(233,46,233),
"VAL":(244,35,244)
}


### Import PDB structure

parser = PDBParser(PERMISSIVE=1)

filename = "./AF-P15823-F1-model_v4.pdb"
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
                atom_xyz = atom.get_coord()
                #Factor = atom.get_bfactor()
                points_3d.append(atom_xyz)
                colorlist.append(resn)

points_3d = np.array(points_3d)
for i in range(len(colorlist)):
    colorlist[i] = AAcolors.get(colorlist[i])
#colorlist = np.array(colorlist, dtype = int)

###### Photography ######

# Define intrinsic camera parameters (currently set for pinhole)
focal_length = 500  # will determine if structure fits, ~zoom level
image_width = 1024
image_height = image_width
intrinsic_matrix = np.array([ 
	[focal_length, 0, image_width/2],  # camera x vector 
	[0, focal_length, image_height/2],  # camera y vector
	[0, 0, 1]  # camera z vector (aim @ PDB centroid?)
])

### Shoot eight images from orthogonal angles

rvec_list = [[0, 0, 0],  # List of ortho pose vectors
             [0, 0, 1],
             [0, 1, 1],
             [1, 1, 1],
             [1, 1, 0],
             [1, 0, 0],
             [0, 1, 0],
             [1, 0, 1]]

# Camera loves you mode
rvec_list = []
for shutter in range(29999):
    rvec_list.append([random.random(), random.random(), random.random()])

for shot in tqdm(range(len(rvec_list))):

# Define extrinsic camera parameters

    rvec = np.array(rvec_list[shot], dtype=np.float32)
    #rvec = np.array([0, 0, 0], dtype=np.float32) 
    tvec = np.array([0, 0, 100], dtype=np.float32)  # x, y, distance from subject

### Project 3D points onto 2D plane 
    points_2d, _ = cv2.projectPoints(points_3d, 
								rvec, tvec.reshape(-1, 1), 
								intrinsic_matrix, 
								None) 

### Plot 2D points as images  
    img = np.zeros([image_height, image_width, 4],  
               dtype=np.uint8)
    kludge = 0
    for point in points_2d.astype(int):
        #color = (100,0,0)
        img = cv2.circle(img, tuple(point[0]), 3, colorlist[kludge], -1)
        kludge += 1
    imname = structure_id + "_" + str(chain) + "_pose" + str(shot) + "_image"
    #cv2.imshow(imname, img) 
    #cv2.waitKey(200)
