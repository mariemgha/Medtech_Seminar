import argparse
from yacs.config import CfgNode as CN
import os.path as osp
import os
import cv2
import numpy as np
from time import time
import pickle
import matplotlib.pyplot as plt
from dataset.annotate import  get_dart_scores, draw #. draw
from scipy.spatial.distance import pdist, squareform

def bboxes_to_xy(bboxes, max_darts=3):
    for i in range(len(bboxes)):
     if np.all(bboxes[i] == 0.0):
      print(i)
    #print(bboxes)
    xy = np.zeros((4 + max_darts, 3), dtype=np.float32)
    for cls in range(5):
        if cls == 0:
            dart_xys = bboxes[bboxes[:, 4] == 0, :2]#[:max_darts]
            print('this: ',dart_xys)
            # Select the most different points
            if len(dart_xys) > max_darts:
                dart_xys = max_distance_triangle(dart_xys)
                print('filtered darts:  ', dart_xys)
            xy[4:4 + len(dart_xys), :2] = dart_xys
        else:
            cal = bboxes[bboxes[:, 4] == cls, :2]
            if len(cal):
                xy[cls - 1, :2] = cal[0]

    # Mark valid points
    xy[(xy[:, 0] > 0) & (xy[:, 1] > 0), -1] = 1
    
    # Check if all calibration points are present
    if np.sum(xy[:4, -1]) == 4:
        return xy
    else:
        xy = est_cal_pts(xy)
    
    return xy

def max_distance_triangle(points):
    from scipy.spatial.distance import pdist, squareform
    dist_matrix = squareform(pdist(points, 'euclidean'))
    max_perimeter = 0
    best_triangle = None
    n = points.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                perimeter = dist_matrix[i, j] + dist_matrix[j, k] + dist_matrix[k, i]
                if perimeter > max_perimeter:
                    max_perimeter = perimeter
                    best_triangle = (i, j, k)
    return  points[list(best_triangle)]


def est_cal_pts(xy):
    print("Estimating calibration points")
    missing_idx = np.where(xy[:4, -1] == 0)[0]
    print("Missing indices:", missing_idx)
    if len(missing_idx) == 1:
        if missing_idx[0] <= 1:
            center = np.mean(xy[2:4, :2], axis=0)
            xy[:, :2] -= center
            if missing_idx[0] == 0:
                xy[0, 0] = -xy[1, 0]
                xy[0, 1] = -xy[1, 1]
                xy[0, 2] = 1
            else:
                xy[1, 0] = -xy[0, 0]
                xy[1, 1] = -xy[0, 1]
                xy[1, 2] = 1
            xy[:, :2] += center
        else:
            center = np.mean(xy[:2, :2], axis=0)
            xy[:, :2] -= center
            if missing_idx[0] == 2:
                xy[2, 0] = -xy[3, 0]
                xy[2, 1] = -xy[3, 1]
                xy[2, 2] = 1
            else:
                xy[3, 0] = -xy[2, 0]
                xy[3, 1] = -xy[2, 1]
                xy[3, 2] = 1
            xy[:, :2] += center
    else:
        print('Missed more than 1 calibration point')
    return xy

# Global variable to store noise points
Noise_points = None


def predict(yolo, img, cfg,dart_bboxes ,filtered_bboxes, max_darts=3, write=True):

    print("entering predict_V1")
    bboxes = yolo.predict(img)
    print('bboxes: ',bboxes)
    #filtered_bboxes = filter_bboxes(bboxes,frame)
    #print('filtered bboxes',filtered_bboxes)
    # Filter the array where the fifth column is 0
    darts_preds= bboxes[bboxes[:, 4] == 0]
    if filtered_bboxes is not None:
        combined_bboxes=np.vstack((dart_bboxes,filtered_bboxes))
    else: 
        combined_bboxes=dart_bboxes
    #print(darts_preds)
    #predict the score only of the darts appearing in the dartboard
    relevant_preds = []
    # Check only the last three arrays in preds, as they are reserved for darts
    for i,pred in enumerate(darts_preds):  
        #print('pred:  ',pred)
        dart_x = pred[0] 
        dart_y = pred[1]
        #match=False
        for j, bbox in enumerate(combined_bboxes):
            x1, y1, x2, y2 = bbox
            if x1-0.1 <= dart_x <= x2+0.1 and y1-0.1 <= dart_y <= y2+0.1:
                print(f"Prediction {i} is within bounding box {j}: {bbox}")
                relevant_preds.append(pred)
                #ämatch=True
                #print('relevant preds:  ', relevant_preds)
                break
        #if match==False:
         #   relevant_preds.append(np.array([0.0, 0.0, 0.0]))
        #if match==False:
         #   relevant_preds.append([0.0,0.0,1.0])
    # Replace the last three predictions with the relevant ones
    bboxes=bboxes[bboxes[:, 4] != 0] 
    print('combined bboxes:  ', combined_bboxes) 
    #print('relevant: ',relevant_preds)
    #print('bboxes:  ',bboxes)
    if relevant_preds:
        bboxes=np.vstack((relevant_preds,bboxes))
    print('modified bboxes: ',bboxes)
    preds = bboxes_to_xy(bboxes, max_darts)
    xy = preds
    print('XY processed:  ',xy)
    print('predicted successfully')
    #xy = preds
    #print('XY: ', xy)
    annotated_image = draw(img.copy(), xy, cfg, circles=True, score=True)
    scores = get_dart_scores(xy, cfg)
    #print('scores:', scores)

    return scores, annotated_image

