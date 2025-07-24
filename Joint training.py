# joint training
# Import the Python libraries
import numpy as np
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import zscore
from bls.processing.replaceNan import replaceNan
from bls.processing.one_hot_m import one_hot_m
from bls.model.bls_incremental_joint import bls_incremental_joint
#from bls.model.bls_train_fscore_incremental import bls_train_fscore_incremental
from math import ceil


seed = 1 # set the seed for generating random numbers
num_class = 6 # number of the class

# Load the datasets
dataset = pd.read_csv('./feature_all1205.csv', quoting=csv.QUOTE_NONE)#Small sample size, 15 samples per label
dataset = np.array(dataset)
#dataset = dataset[: 36000, :]
train_dataset, test_dataset = train_test_split(dataset, test_size=0.2)

# Normalize training data
train_x = train_dataset[:, 0:train_dataset.shape[1] - 1]
train_x = zscore(train_x, axis=0, ddof=1)
replaceNan(train_x)
train_y = train_dataset[:, train_dataset.shape[1] - 1: train_dataset.shape[1]]

'''
output_features = np.copy(train_x)
output_labels = np.copy(train_y)

for i in range(1):
    rotated_features = np.copy(train_x)
    for j in range(len(rotated_features)):
        feature = rotated_features[j]
        angle = np.random.uniform(-30, 30)
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        feature[:2] = np.dot(feature[:2], rotation_matrix)
        rotated_features[j] = feature
    output_features = np.vstack((output_features, rotated_features))
    output_labels = np.vstack((output_labels, train_y))

train_x = output_features
train_y = output_labels
'''

# Change training labels
inds1 = np.where(train_y == 0) #?????????0????
train_y[inds1] = 6 #????????0?????15?

# Normalize test data
test_x = test_dataset[:, 0:test_dataset.shape[1] - 1]
test_x = zscore(test_x, axis=0, ddof=1)  # For each feature, mean = 0 and std = 1
replaceNan(test_x)							 # Replace "nan" with 0
test_y = test_dataset[:, test_dataset.shape[1] - 1: test_dataset.shape[1]]

# Change test labels
inds1 = np.where(test_y == 0)
test_y[inds1] = 6

train_y = one_hot_m(train_y, num_class)
test_y = one_hot_m(test_y, num_class)

# BLS parameters
C = 2**-28 # parameter for sparse regularization???????
s = 0.8     # the shrinkage parameter for enhancement nodes ?????????

# N1* - the number of mapped feature nodes
# N2* - the groups of mapped features
# N3* - the number of enhancement nodes

N1_bls = 8
N2_bls = 20
N3_bls = 600


epochs = 1 # number of epochs
inputData = ceil(train_x.shape[0]*0.05)

# BLS parameters for incremental learning
l = 20    # steps
m2 = 40  # 20,40, additional enhancement nodes for each step

train_xf = train_x # the entire training dataset
train_yf = train_y # the entire training labels

train_x = train_xf[:(int)(inputData), :] # training data at the beginning of the incremental learning
train_y = train_yf[:(int)(inputData), :] # training labels at the beginning of the incremental learning

m = int(ceil((train_xf.shape[0] - inputData) / l)) # the number of added data points/step

print("Incremental step is: ", l)

train_err = np.zeros((1, epochs))
test_err = np.zeros((1, epochs))
train_time = np.zeros((1, epochs))
test_time = np.zeros((1, epochs))

# # BLS ----------------------------------------------------------------
print("================== BLS (incremental)===========================\n\n")

#np.random.seed(seed) # set the seed for generating random numbers
for j in range(0, epochs):
    TrainingAccuracy, TestingAccuracy, Training_time, Testing_time, f_score, pre, recall = \
	bls_incremental_joint(train_x, train_y, train_xf, train_yf, test_x, test_y, s, C,
								 N1_bls, N2_bls, N3_bls, inputData, m, m2, l)
    print ("BLS Train Acc: ", TrainingAccuracy*100, "% BLS Test Acc: ", TestingAccuracy*100,
	   "%\nfscore: ", f_score, "  Precision: ", pre, "  Recall: ", recall,
       "\nTraining time: ", Training_time, "s Testing time: " ,Testing_time, "s")

    print("End of the execution")
