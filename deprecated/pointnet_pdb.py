"""
Title: Point cloud classification with PointNet
Author: [David Griffiths](https://dgriffiths3.github.io)
Date created: 2020/05/25
Last modified: 2020/05/26
Description: Implementation of PointNet for ModelNet10 classification.
Accelerator: GPU
"""
"""
# Point cloud classification
"""

"""
## Introduction

Classification, detection and segmentation of unordered 3D point sets i.e. point clouds
is a core problem in computer vision. This example implements the seminal point cloud
deep learning paper [PointNet (Qi et al., 2017)](https://arxiv.org/abs/1612.00593). For a
detailed intoduction on PointNet see [this blog
post](https://medium.com/@luis_gonzales/an-in-depth-look-at-pointnet-111d7efdaa1a).
"""

"""
## Setup

If using colab first install trimesh with `!pip install trimesh`.
"""


import os
import glob
import sys
from turtle import shape
#import trimesh
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from Bio.PDB.PDBParser import PDBParser
from tqdm import tqdm
import csv

tf.random.set_seed(1234)

DATA_DIR = "./images"
captions = {}

with open("captions.csv", mode="r") as inp:
    reader = csv.reader(inp)
    captions = {rows[0]: rows[1] for rows in reader}

"""
## Load dataset

We use the ModelNet10 model dataset, the smaller 10 class version of the ModelNet40
dataset. First download the data:


DATA_DIR = tf.keras.utils.get_file(
    "modelnet.zip",
    "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip",
    extract=True,
)
DATA_DIR = os.path.join(os.path.dirname(DATA_DIR), "ModelNet10")


We can use the `trimesh` package to read and visualize the `.off` mesh files.


#mesh = trimesh.load(os.path.join(DATA_DIR, "chair/train/chair_0001.off"))
#mesh.show()


To convert a mesh file to a point cloud we first need to sample points on the mesh
surface. `.sample()` performs a unifrom random sampling. Here we sample at 2048 locations
and visualize in `matplotlib`.


#points = mesh.sample(2048)
#points = 

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(points[:, 0], points[:, 1], points[:, 2])
ax.set_axis_off()
plt.show()


To generate a `tf.data.Dataset()` we need to first parse through the ModelNet data
folders. Each mesh is loaded and sampled into a point cloud before being added to a
standard python list and converted to a `numpy` array. We also store the current
enumerate index value as the object label and use a dictionary to recall this later.
"""

def get_max_pdb_len(pdbfile):
    # Using readline()
    maxlen = 0
    with open(pdbfile) as pdb:
        while True:
            line = pdb.readline()
            #print(line[0:4])
    
            if not line:
                break
            if line[0:4] == "ATOM":
                maxlen += 1
    
    return maxlen


def parse_dataset(maxpad):
    train_points = []
    train_labels = []
    test_points = []
    test_labels = []
    class_map = {}
    folders = glob.glob(os.path.join(DATA_DIR))

    parser = PDBParser(PERMISSIVE=1)

    for i, folder in enumerate(folders):
        print("processing class: {}".format(os.path.basename(folder)))
        # store folder name with ID so we can retrieve later
        # gather all files
        train_files = glob.glob(os.path.join(folder, "train/*"))
        test_files = glob.glob(os.path.join(folder, "test/*"))

        for f in tqdm(train_files):
            ### Import PDB structure
            #filename = "./AF-P42212-F1-model_v4.pdb"
            structure_id = f.split("-")[1]
            structure = parser.get_structure(structure_id, f)
            class_map[i] = captions[structure_id]  # first mfGO annotation
            # Extract atomic 3d coordinates array
            points_3d = np.zeros([maxpad,3])
            idx = 0
            for model in structure:
                for chain in model:    
                    for residue in chain:
                        #resn = residue.get_resname()
                        for atom in residue:
                            #atom_xyz = atom.get_vector()
                            points_3d[idx] = atom.get_vector()
                            idx += 1

            train_points = np.concatenate(train_points, points_3d)
            train_labels = np.concatenate(train_labels, class_map[i])

        for f in tqdm(test_files):
            ### Import PDB structure
            #filename = "./AF-P42212-F1-model_v4.pdb"
            structure_id = f.split("-")[1]
            structure = parser.get_structure(structure_id, f)
            class_map[i] = captions[structure_id]  # first mfGO annotation

            # Extract atomic 3d coordinates array
            points_3d = np.zeros([maxpad,3])
            idx = 0
            for model in structure:
                for chain in model:
                    points_3d = np.array([])
                    for residue in chain:
                        #resn = residue.get_resname()
                        for atom in residue:
                            #atom_xyz = atom.get_vector()
                            points_3d[idx] = atom.get_vector()
                            idx += 1
            
            test_points = np.concatenate(test_points, points_3d)
            test_labels = np.concatenate(test_labels, class_map[i])

    #print(train_points)

    return (
        train_points.astype(np.float32),
        test_points.astype(np.float32),
        train_labels.astype(np.float32),
        test_labels.astype(np.float32),
        class_map
    )


"""
Set the number of points to sample and batch size and parse the dataset. This can take
~5minutes to complete.
"""

# Find maximum length of PDB set and calculate padding value
maxpad = 0
for folder in glob.glob(os.path.join(DATA_DIR)):
    for file in tqdm(glob.glob("*.pdb")):
        if get_max_pdb_len(file) > maxpad:
            maxpad = get_max_pdb_len(file)
print("Maxmimum number of atoms in PDB file set: ", maxpad)
print("Extracted atomic coordinate arrays will be padded to length", maxpad, "with [0,0,0].")

desc = np.array(captions[1])
NUM_CLASSES = len(np.unique(desc))

#NUM_CLASSES = 100
BATCH_SIZE = 32

train_points, test_points, train_labels, test_labels, CLASS_MAP = parse_dataset(maxpad)


"""
Our data can now be read into a `tf.data.Dataset()` object. We set the shuffle buffer
size to the entire size of the dataset as prior to this the data is ordered by class.
Data augmentation is important when working with point cloud data. We create a
augmentation function to jitter and shuffle the train dataset.
"""


def augment(points, label):
    # jitter points
    points += tf.random.uniform(points.shape, -0.005, 0.005, dtype=tf.float64)
    # shuffle points
    points = tf.random.shuffle(points)
    return points, label


train_dataset = tf.data.Dataset.from_tensor_slices((train_points, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_points, test_labels))

train_dataset = train_dataset.shuffle(len(train_points)).map(augment).batch(BATCH_SIZE)
test_dataset = test_dataset.shuffle(len(test_points)).batch(BATCH_SIZE)

"""
### Build a model

Each convolution and fully-connected layer (with exception for end layers) consits of
Convolution / Dense -> Batch Normalization -> ReLU Activation.
"""


def conv_bn(x, filters):
    x = layers.Conv1D(filters, kernel_size=1, padding="valid")(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)


def dense_bn(x, filters):
    x = layers.Dense(filters)(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)


"""
PointNet consists of two core components. The primary MLP network, and the transformer
net (T-net). The T-net aims to learn an affine transformation matrix by its own mini
network. The T-net is used twice. The first time to transform the input features (n, 3)
into a canonical representation. The second is an affine transformation for alignment in
feature space (n, 3). As per the original paper we constrain the transformation to be
close to an orthogonal matrix (i.e. ||X*X^T - I|| = 0).
"""


class OrthogonalRegularizer(keras.regularizers.Regularizer):
    def __init__(self, num_features, l2reg=0.001):
        self.num_features = num_features
        self.l2reg = l2reg
        self.eye = tf.eye(num_features)

    def __call__(self, x):
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))


"""
 We can then define a general function to build T-net layers.
"""


def tnet(inputs, num_features):
    # Initalise bias as the indentity matrix
    bias = keras.initializers.Constant(np.eye(num_features).flatten())
    reg = OrthogonalRegularizer(num_features)

    x = conv_bn(inputs, 32)
    x = conv_bn(x, 64)
    x = conv_bn(x, 512)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, 256)
    x = dense_bn(x, 128)
    x = layers.Dense(
        num_features * num_features,
        kernel_initializer="zeros",
        bias_initializer=bias,
        activity_regularizer=reg,
    )(x)
    feat_T = layers.Reshape((num_features, num_features))(x)
    # Apply affine transformation to input features
    return layers.Dot(axes=(2, 1))([inputs, feat_T])


"""
The main network can be then implemented in the same manner where the t-net mini models
can be dropped in a layers in the graph. Here we replicate the network architecture
published in the original paper but with half the number of weights at each layer as we
are using the smaller 10 class ModelNet dataset.
"""

inputs = keras.Input(shape=(maxpad, 3))

x = tnet(inputs, 3)
x = conv_bn(x, 32)
x = conv_bn(x, 32)
x = tnet(x, 32)
x = conv_bn(x, 32)
x = conv_bn(x, 64)
x = conv_bn(x, 512)
x = layers.GlobalMaxPooling1D()(x)
x = dense_bn(x, 256)
x = layers.Dropout(0.3)(x)
x = dense_bn(x, 128)
x = layers.Dropout(0.3)(x)

outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="pointnet")
model.summary()

"""
### Train model

Once the model is defined it can be trained like any other standard classification model
using `.compile()` and `.fit()`.
"""

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["sparse_categorical_accuracy"],
)

model.fit(train_dataset, epochs=20, validation_data=test_dataset)

"""
## Visualize predictions

We can use matplotlib to visualize our trained model performance.
"""

data = test_dataset.take(1)

points, labels = list(data)[0]
points = points[:8, ...]
labels = labels[:8, ...]

# run test data through model
preds = model.predict(points)
preds = tf.math.argmax(preds, -1)

points = points.numpy()

# plot points with predicted class and label
fig = plt.figure(figsize=(15, 10))
for i in range(8):
    ax = fig.add_subplot(2, 4, i + 1, projection="3d")
    ax.scatter(points[i, :, 0], points[i, :, 1], points[i, :, 2])
    ax.set_title(
        "pred: {:}, label: {:}".format(
            CLASS_MAP[preds[i].numpy()], CLASS_MAP[labels.numpy()[i]]
        )
    )
    ax.set_axis_off()
plt.show()
