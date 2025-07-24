# Fine-tuning
# Import the Python libraries
import time
import random
import numpy as np
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import zscore
from bls.processing.replaceNan import replaceNan
from bls.processing.one_hot_m import one_hot_m
from bls.model.bls_incremental_frozen import bls_incremental_frozen
from bls.model.bls_incremental_unfrozen import bls_incremental_unfrozen
from math import ceil

def get_random_data(dataset):
    num_rows, num_cols = dataset.shape
    random_indices = random.sample(range(num_rows), 2400)
    random_data = dataset[random_indices, :]
    return random_data

seed = 1 # set the seed for generating random numbers
num_class = 6 # number of the class

# Load the datasets
dataset0 = pd.read_csv('./feature_all1205_AMC3.csv', quoting=csv.QUOTE_NONE)#Small sample size, 15 samples per label
dataset = pd.read_csv('./feature_all1205.csv', quoting=csv.QUOTE_NONE)#Small sample size, 15 samples per label
dataset0 = np.array(dataset0)
dataset = np.array(dataset)

train_dataset0, test_dataset0 = train_test_split(dataset0, test_size=0.2)
train_dataset, test_dataset = train_test_split(dataset, test_size=0.2)

# Normalize training data
train_x0 = train_dataset0[:, 0:train_dataset0.shape[1] - 1] #??train_dataset?????2????????????????????
train_x0 = zscore(train_x0, axis=0, ddof=1) # ?????????????????????0?????1?
replaceNan(train_x0)                           # Replace "nan" with 0
train_y0 = train_dataset0[:, train_dataset0.shape[1] - 1: train_dataset0.shape[1]] #??train_dataset????????????

# Change training labels
inds1 = np.where(train_y0 == 0) #?????????0????
train_y0[inds1] = 6 #????????0?????15?

train_y0 = one_hot_m(train_y0, num_class)

test_x0 = train_x0  # 假设使用同样的数据进行测试
test_y0 = train_y0

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

# Normalize training data
train_x = train_dataset[:, 0:train_dataset.shape[1] - 1] #??train_dataset?????2????????????????????
train_x = zscore(train_x, axis=0, ddof=1) # ?????????????????????0?????1?
replaceNan(train_x)                           # Replace "nan" with 0
train_y = train_dataset[:, train_dataset.shape[1] - 1: train_dataset.shape[1]] #??train_dataset????????????

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

# BLS parameters for incremental learning
l = 20    # steps
m2 = 40  # 20,40, additional enhancement nodes for each step

train_xf = train_x0 # the entire training dataset
train_yf = train_y0 # the entire training labels

inputData = ceil(train_x0.shape[0]*0.05)
m = int(ceil((train_xf.shape[0] - inputData) / l)) # the number of added data points/step

print("Incremental step is: ", l)

train_err = np.zeros((1, epochs))
test_err = np.zeros((1, epochs))
train_time = np.zeros((1, epochs))
test_time = np.zeros((1, epochs))

# # BLS ----------------------------------------------------------------
print("================== BLS (incremental)  pre-training ===========================\n\n")

np.random.seed(seed) # set the seed for generating random numbers
for j in range(0, epochs):
    TestingAccuracy, f_score, train_err, test_err, Training_time, Testing_time = \
    bls_incremental_frozen(train_x0, train_y0, train_xf, train_yf, test_x, test_y, s, C,
                    N1_bls, N2_bls, N3_bls, inputData, m, m2, l)



print ("BLS Train err: ", train_err, "% BLS Test Acc: ", TestingAccuracy*100,
	   "%\nfscore: ", f_score, "\nTraining time: ", Training_time, "s Testing time: " ,Testing_time, "s")



inputData = ceil(train_x.shape[0]*0.05)
train_x0 = train_xf[:(int)(inputData), :] # training data at the beginning of the incremental learning
train_y0 = train_yf[:(int)(inputData), :] # training labels at the beginning of the incremental learning
m = int(ceil((train_xf.shape[0] - inputData) / l)) # the number of added data points/step
train_xf = train_x # the entire training dataset
train_yf = train_y # the entire training labels

print("Incremental step is: ", l)

train_err = np.zeros((1, epochs))
test_err = np.zeros((1, epochs))
train_time = np.zeros((1, epochs))
test_time = np.zeros((1, epochs))

# # BLS ----------------------------------------------------------------
print("================== BLS (incremental)  fine-tuning ===========================\n\n")

np.random.seed(seed) # set the seed for generating random numbers
for j in range(0, epochs):
    TestingAccuracy_z, f_score, pre, recall, train_err, test_err, Training_time, Testing_time = \
	bls_incremental_unfrozen(train_x, train_y, train_xf, train_yf, test_x, test_y, s, C,
								 N1_bls, N2_bls, N3_bls, inputData, m, m2, l)


print ("BLS Test Acc: ", TestingAccuracy_z*100,
	   "%\nfscore: ", f_score, "  Precision: ", pre, "  Recall: ", recall,
       "\nTraining time: ", Training_time, "s Testing time: " ,Testing_time, "s")

print("End of the execution")


