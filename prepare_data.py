#!/usr/bin/env python3
"""
Script for preparing data for the RSNA Pneumonia Detection Challenge
"""
import json
import glob
import os
from collections import defaultdict
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import util

sz = 1000
with open('settings.json') as json_data_file:
    json_data = json.load(json_data_file)


def getpath(pid):
    """ Given an ID, return the path of the jpeg relative to the data dir """
    return os.path.basename(json_data["TRAIN_JPG_DIR"])+"/"+pids[pid]+".jpg"


def anno(idx):
    """ Return a list of bounding box annotations given a list of IDs """
    annolist = []
    for p_id in idx:
        boxes = boxdict[pids[p_id]]
        if boxes:
            for box in boxes:
                annolist.append([getpath(p_id), box[0], box[1],
                                 box[2], box[3], 'pneumonia'])
        else:
            annolist.append([getpath(p_id), '', '', '', '', ''])
    return annolist


df = pd.read_csv(json_data['RAW_TRAIN_LABELS'], keep_default_na=False)

boxdict = defaultdict(list)
pids = []
targets = []
for index, row in df.iterrows():
    pid = row['patientId']
    x = row['x']
    y = row['y']
    width = row['width']
    height = row['height']
    target = int(row['Target'])
    conversion = sz/1000
    if x != '':
        x1 = int(float(x))
        y1 = int(float(y))
        x2 = x1+int(float(width))-1
        if x2 > sz:
            x2 = sz
        y2 = y1+int(float(height))-1
        if y2 > sz:
            y2 = sz

        x1 = int(x1*conversion)
        y1 = int(y1*conversion)
        x2 = int(x2*conversion)
        y2 = int(y2*conversion)
        target = 'pneumonia'

        boxdict[pid].append((x1, y1, x2, y2))

    if pid not in pids:
        pids.append(pid)
        targets.append(target)

stratSplit = StratifiedKFold(n_splits=20, shuffle=True, random_state=42)
folds = list(stratSplit.split(pids, targets))

# I trained with a few folds, this one yielded the best models
(train_idx, val_idx) = folds[1]

train_annotations = anno(train_idx)
val_annotations = anno(val_idx)
print(f"Training: {len(train_idx)}, validation: {len(val_idx)}")
pd_train_annotations = pd.DataFrame.from_records(train_annotations)
pd_val_annotations = pd.DataFrame.from_records(val_annotations)

processed_dir = os.path.dirname(json_data["TRAIN_CSV"])
if not os.path.exists(processed_dir):
    os.mkdir(processed_dir)

pd_train_annotations.to_csv(json_data["TRAIN_CSV"], index=False, header=False)
pd_val_annotations.to_csv(json_data["VAL_CSV"], index=False, header=False)
with open(json_data["CLASSES_CSV"], 'w') as class_file:
    class_file.write('pneumonia,0')


train_jpg_dir = json_data["TRAIN_JPG_DIR"]
if not os.path.exists(train_jpg_dir):
    os.mkdir(train_jpg_dir)

    for i, dcm_file in enumerate(glob.glob(os.path.join(json_data["TRAIN_DCM_DIR"], "*.dcm"))):
        bn = os.path.basename(dcm_file)
        out_file = os.path.join(train_jpg_dir, bn[:-4]+".jpg")
        print(f"Converting training image # {i+1}\r", end="")
        util.dicom_to_jpg(dcm_file, out_file, sz)
print()
test_jpg_dir = json_data["TEST_JPG_DIR"]
if not os.path.exists(test_jpg_dir):
    os.mkdir(test_jpg_dir)

    for i, dcm_file in enumerate(glob.glob(os.path.join(json_data["TEST_DCM_DIR"], "*.dcm"))):
        bn = os.path.basename(dcm_file)
        out_file = os.path.join(test_jpg_dir, bn[:-4]+".jpg")
        print(f"Converting test image # {i+1}\r", end="")
        util.dicom_to_jpg(dcm_file, out_file, sz)
print()
