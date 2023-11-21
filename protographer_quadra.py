########################################
#  Converts PDBs to quadra-holograms   #
#                                      #
#  Usage:                              #
#  script infile.pdb out_dir/ n_imgs   #
########################################

from Bio.PDB.PDBParser import PDBParser
import numpy as np
import cv2
import os
import sys
import random
from tqdm import tqdm
import wget

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

def download_af_pdb(filename, out):
    if filename in os.listdir(out):
        print(filename, "already in", out, "; skipping download.")
    else:
        url = "https://alphafold.ebi.ac.uk/files/" + filename
        filename = wget.download(url, out=out)
        print("Fetched:", out + filename)
    return


def get_pdb_points(filename):
    '''
    Inputs: 
    filename        <str> an alphafold pdb filename
    
    Returns:
    structure_id    <str> UniProt ID
    points_3d       <nd.array> (3,N) atomic xyz coordinates
    colorlist       <list> (3,N) B,G,R [0-255] resn color values per point_3d
    '''
    parser = PDBParser(PERMISSIVE=1)

    structure_id = filename.split("-")[1]  # assumes alphafold model
    structure = parser.get_structure(structure_id, filename)

    # Extract atomic 3d coordinates array
    for model in structure:
        points_3d = []
        colorlist = []
        for chain in model:
            for residue in chain:
                resn = residue.get_resname()
                for atom in residue:
                    if atom.get_name() in ["N", "C", "CA", "CB"]:
                        atom_xyz = atom.get_coord()
                        points_3d.append(atom_xyz)
                        colorlist.append(resn)
    points_3d = np.array(points_3d)

    # Map colors by amino acid
    for i in range(len(colorlist)):
        colorlist[i] = AAcolors.get(colorlist[i])
    #colorlist = np.array(colorlist, dtype = int)
    return structure_id, points_3d, colorlist


def pdb_take_photo(points_3d, colorlist, rvec, wiggle):
    '''
    Inputs: 
    points_3d       <nd.array> (3,N) atomic xyz coordinates
    colorlist       <list> (3,N) B,G,R [0-255] resn color values per point_3d
    rvec            <list> (3,1) camera rotational vector (POV angle)
    
    Returns:
    img             <nd.array> (512,512,3) 512px BGR color image
    '''
    # Define intrinsic camera parameters (currently set for pinhole)
    focal_length = 500  # will determine if structure fits, ~zoom level
    image_width = 512
    image_height = image_width  # square
    intrinsic_matrix = np.array([ 
        [focal_length, 0, image_width/2],  # camera x vector 
        [0, focal_length, image_height/2],  # camera y vector
        [0, 0, 1]  # camera z vector (aim @ PDB centroid?)
    ])

    # Apply random offset in xyz to rvec
    rvec = [rvec[0] - wiggle,
            rvec[1] - wiggle,
            rvec[2] + wiggle]

    # Define extrinsic camera parameters
    tvec = np.array([0, 0, 120], dtype=np.float32)  # x, y, zoom
    rvec = np.array(rvec, dtype=np.float32)

    ### Project 3D points onto 2D plane 
    points_2d, _ = cv2.projectPoints(points_3d, 
                                rvec, tvec.reshape(-1, 1), 
                                intrinsic_matrix, 
                                None) 

    ### Plot 2D points as L/R images  
    img = np.zeros([image_height, image_width, 3], dtype=np.uint8)
    ptSize = 1  # Size of amino acid points
    fill = -1  # -1 = filled, else stroke width in px
    k = 0
    for point in points_2d.astype(int):
        img = cv2.circle(img, tuple(point[0]), ptSize, colorlist[k], fill)
        k += 1

    #imname = structure_id + "_pose_" + str(rvec)
    
    #cv2.imshow(imname, img)
    #cv2.waitKey(8)
    #cv2.destroyAllWindows()

    return img

pdb = sys.argv[1]
out = sys.argv[2]
wigrange = 0.02
photos = 1000

download_af_pdb(pdb, out)
filename = out + pdb
structure_id, points_3d, colorlist = get_pdb_points(filename)

r_vecs = [[1, 1, 1],  # "Front"
        [1, -1, 1],   # "Top"
        [-1, 1, 1],   # "Right"
        [1, 1, -1]]   # "Bottom"

for round in tqdm(range(photos)):
    wiggle = random.uniform(0-wigrange, wigrange)
    images = []
    for photo in r_vecs:
        images.append(pdb_take_photo(points_3d, colorlist, photo, wiggle))

    img_ab = cv2.hconcat([images[0], images[1]])
    img_cd = cv2.hconcat([images[2], images[3]])
    quadra = cv2.vconcat([img_ab, img_cd])
    imname = structure_id + "_pose_" + str(round)

    cv2.imshow(imname, quadra)
    cv2.waitKey(2)
    cv2.destroyAllWindows()

    #cv2.imwrite("./swiss/focal500_tran120/" + imname + ".png", quadra)