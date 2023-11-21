from Bio.PDB.PDBParser import PDBParser
import numpy as np
import cv2
import csv
import random
from tqdm import tqdm
from matplotlib import pyplot as plt
import time

###### BGR amino acid encoding table
# Chosen from Color Alphabet published in
# Colour: Design & Creativity (5) (2010): 10, 1-23

AAcolors = {
"ALA":(255,163,240),
"ARG":(220,117,0),
"ASN":(0,63,153),
"ASP":(72,206,43),
"CYS":(153,204,255),
"GLN":(181,255,148),
"GLU":(0,204,157),
"GLY":(136,0,197),
"HIS":(5,164,255),
"ILE":(187,168,255),
"LEU":(0,102,66),
"LYS":(16,0,255),
"MET":(242,241,94),
"PHE":(143,153,0),
"PRO":(102,255,224),
"SER":(255,10,116),
"THR":(0,0,153),
"TRP":(128,255,255),
"TYR":(0,255,255),
"VAL":(5,80,255)
}

### Import PDB structure

parser = PDBParser(PERMISSIVE=1)

filename = "./AF-P02699-F1-model_v4.pdb"
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
                if atom.get_name() == "CA":
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
image_width = 1200
image_height = image_width
intrinsic_matrix = np.array([ 
	[focal_length, 0, image_width/2],  # camera x vector 
	[0, focal_length, image_height/2],  # camera y vector
	[0, 0, 1]  # camera z vector (aim @ PDB centroid?)
])

# Camera loves you mode
q = 0.5  # stereoscopic distance factor
rvec_list = []
for shutter in range(9):
    rvec_list.append([random.uniform(-1, 1-q),  # 0.75 allowance for R
          random.uniform(-1, 1-q),
          random.uniform(-1, 1-q)])

# Right stereoscopic view
rvec_listR = [[coord+q for coord in row] for row in rvec_list]

for shot in tqdm(range(len(rvec_list))):

# Define extrinsic camera parameters
    tvec = np.array([0, 0, 90], dtype=np.float32)  # x, y, zoom
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
    img = np.zeros([image_height, image_width, 1],  
               dtype=np.uint8)
    imgR = np.zeros([image_height, image_width, 1],  
               dtype=np.uint8)
    k = 0
    for point in points_2d.astype(int):
        img = cv2.circle(img, tuple(point[0]), 1, colorlist[k], -1)
        k += 1
    k = 0
    for point in points_2dR.astype(int):
        imgR = cv2.circle(imgR, tuple(point[0]), 1, colorlist[k], -1)
        k += 1
    
    imname = structure_id + "_pose" + str(shot) + "_stereo_image"
    #stereo = cv2.hconcat([img, imgR])
    #cv2.imshow(imname, stereo)
    #cv2.waitKey(1)
    #cv2.destroyAllWindows()

    RT = np.zeros((3,4))
    RT[:3, :3] = np.array([0,0,0], dtype=np.float32)  # Arbitrary rvec L
    RT[:3, 3] = tvec.transpose()
    left_projection = np.dot(intrinsic_matrix, RT)

    RT = np.zeros((3,4))
    RT[:3, :3] = np.array([q,q,q], dtype=np.float32)  # Arbitrary rvec R (L+q)
    RT[:3, 3] = tvec.transpose()
    right_projection = np.dot(intrinsic_matrix, RT)

    triangulation = cv2.triangulatePoints(left_projection, right_projection, points_2d, points_2dR)
    homog_points = triangulation.transpose()

    euclid_points = cv2.convertPointsFromHomogeneous(homog_points)

    if shutter == 8:
        print(homog_points)
        print(euclid_points)
        
    
#    numDisparities = k*4
#    stereoObj = cv2.StereoBM.create(numDisparities=128, blockSize=5)
#    disparity = stereoObj.compute(img,imgR)
#    cv.reprojectImageTo3D(disparity, 3dpoints, Q)
#    plt.imshow(disparity,'gray')
#    plt.show()
##disparity.tofile('data2.csv', sep = ',')
