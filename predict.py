#!/usr/bin/env python3
"""
Script for predicting bounding boxes for the RSNA pneumonia detection challenge
by Phillip Cheng, MD MS
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd

# import utility functions
import util

# This is a modified version of keras-retinanet 0.4.1
# which includes a score metric to estimate the RSNA score
# at the threshold giving the maximum Youden index.
sys.path.append("keras-retinanet")
from keras_retinanet import models


with open('settings.json') as json_data_file:
    json_data = json.load(json_data_file)
model1_path = json_data["MODEL_50"]
model1 = models.load_model(model1_path, backbone_name='resnet50', convert=True, nms=False)

model2_path = json_data["MODEL_101"]
model2 = models.load_model(model2_path, backbone_name='resnet101', convert=True, nms=False)

test_jpg_dir = json_data["TEST_JPG_DIR"]
submission_dir = json_data["SUBMISSION_DIR"]

sz = 224

# threshold for non-max-suppresion for each model
nms_threshold = 0

# shrink bounding box dimensions by this factor, improves test set performance
shrink_factor = 0.17

# threshold for judging overlap of bounding boxes between different networks (for weighted average)
wt_overlap = 0

# threshold for including boxes from model 1
score_threshold1 = 0.04

# threshold for including boxes from model 2
score_threshold2 = 0.05

# threshold for including isolated boxes from either model
solo_min = 0.15

test_ids = []
test_outputs = []

start = time.time()

for i, fname in enumerate(os.listdir(test_jpg_dir)):
    print(f"Predicting boxes for image # {i+1}\r", end="")
    fpath = os.path.join(test_jpg_dir, fname)
    fid = fname[:-4]

    boxes_pred1, scores1 = util.get_detection_from_file(fpath, model1, sz)
    boxes_pred2, scores2 = util.get_detection_from_file(fpath, model2, sz)

    indices1 = np.where(scores1 > score_threshold1)[0]
    scores1 = scores1[indices1]
    boxes_pred1 = boxes_pred1[indices1]
    boxes_pred1, scores1 = util.nms(boxes_pred1, scores1, nms_threshold)

    indices2 = np.where(scores2 > score_threshold2)[0]
    scores2 = scores2[indices2]
    boxes_pred2 = boxes_pred2[indices2]
    boxes_pred2, scores2 = util.nms(boxes_pred2, scores2, nms_threshold)

    boxes_pred = np.concatenate((boxes_pred1, boxes_pred2))
    scores = np.concatenate((scores1, scores2))

    boxes_pred, scores = util.averages(
        boxes_pred, scores, wt_overlap, solo_min)
    util.shrink(boxes_pred, shrink_factor)

    output = ''
    for j, bb in enumerate(boxes_pred):
        x1 = int(bb[0])
        y1 = int(bb[1])
        w = int(bb[2]-x1+1)
        h = int(bb[3]-y1+1)
        output += f'{scores[j]:.3f} {x1} {y1} {w} {h} '
    test_ids.append(fid)
    test_outputs.append(output)
print()
end = time.time()
# print execution time
print(f"Elapsed time = {end-start:.3f} seconds")

test_df = pd.DataFrame({'patientId': test_ids, 'PredictionString': test_outputs},
                       columns=['patientId', 'PredictionString'])
if not os.path.exists(submission_dir):
    os.mkdir(submission_dir)
test_df.to_csv(os.path.join(submission_dir, "stage2_test.csv"), index = False)
