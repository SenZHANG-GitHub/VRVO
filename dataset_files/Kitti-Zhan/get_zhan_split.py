import numpy as np
import os 
import pdb
import random

seed = 1729
random.seed(seed)
np.random.seed(seed)

train_only_seq = [
    ("03", "2011_09_26_drive_0067", 0, 800)
]
train_val_seq = [
    ("00", "2011_10_03_drive_0027", 0, 4540),
    ("01", "2011_10_03_drive_0042", 0, 1100),
    ("02", "2011_10_03_drive_0034", 0, 4660),
    ("04", "2011_09_30_drive_0016", 0, 270),
    ("05", "2011_09_30_drive_0018", 0, 2760),
    ("06", "2011_09_30_drive_0020", 0, 1100),
    ("07", "2011_09_30_drive_0027", 0, 1100),
    ("08", "2011_09_30_drive_0028", 1100, 5170),
]

test_seq = [
    ("09", "2011_09_30_drive_0033", 0, 1590),
    ("10", "2011_09_30_drive_0034", 0, 1200)
]

## Saving the test data
test_data = []
for datum in test_seq:
    seq_name = datum[1]
    date = seq_name.split("drive")[0][:-1]
    folder = "{}/{}_sync".format(date, seq_name)
    for i in range(datum[2], datum[3] + 1):
        line = "{} {:010d} l\n".format(folder, i)
        test_data.append(line)

num_test = len(test_data)
print("=> number of test data: {}".format(num_test)) # 2792

with open("test.txt", "w") as f:
    for line in test_data:
        f.write(line)

## Saving the training and val data
# NOTE that seq 03 is only used for training
train_only_data = []
for datum in train_only_seq:
    seq_name = datum[1]
    date = seq_name.split("drive")[0][:-1]
    folder = "{}/{}_sync".format(date, seq_name)
    for i in range(datum[2]+1, datum[3]):
        line = "{} {:010d} l\n".format(folder, i)
        train_only_data.append(line)

train_val_data = []
for datum in train_val_seq:
    seq_name = datum[1]
    date = seq_name.split("drive")[0][:-1]
    folder = "{}/{}_sync".format(date, seq_name)
    for i in range(datum[2]+1, datum[3]):
        line = "{} {:010d} l\n".format(folder, i)
        train_val_data.append(line)

num_all = len(train_only_data) + len(train_val_data)
num_train = int(num_all * 22537 / (22537 + 888))
num_val = num_all - num_train
print("=> total number of data loaded for train/val: {}".format(num_all)) # 20391
print("=> number of training data: {}".format(num_train)) # 19618
print("=> number of val data: {}".format(num_val)) # 773

random.shuffle(train_only_data)
random.shuffle(train_val_data)

val_data = train_val_data[:num_val]
train_data = train_val_data[num_val:] + train_only_data

with open("train.txt", "w") as f:
    for line in train_data:
        f.write(line)
with open("val.txt", "w") as f:
    for line in val_data:
        f.write(line)
