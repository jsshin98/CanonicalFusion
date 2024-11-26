import numpy as np
import trimesh
import cv2
import torch
from pysdf import SDF
from skimage import measure
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

def draw_keypoints_per_person(img, all_keypoints, all_scores, confs,
                              keypoint_threshold=2, conf_threshold=0.9):
    # Example: keypoints_img =
    # draw_keypoints_per_person(img, output["keypoints"],
    # output["keypoints_scores"], output["scores"], keypoint_threshold=2)
    cmap = plt.get_cmap('rainbow')
    img_copy = img.copy()
    color_id = np.arange(1, 255, 255 // len(all_keypoints)).tolist()[::-1]
    for person_id in range(len(all_keypoints)):
        if confs[person_id] > conf_threshold:
            keypoints = all_keypoints[person_id, ...]
            scores = all_scores[person_id, ...]
            for kp in range(len(scores)):
                if scores[kp] > keypoint_threshold:
                    keypoint = tuple(map(int, keypoints[kp, :2].detach().numpy().tolist()))
                    color = tuple(np.asarray(cmap(color_id[person_id])[:-1]) * 255)
                    cv2.circle(img_copy, keypoint, 30, color, -1)

    return img_copy

def get_limbs_from_keypoints(keypoints):
    # Example: limbs = get_limbs_from_keypoints(keypoints)
    limbs = [[keypoints.index('right_eye'), keypoints.index('nose')],
             [keypoints.index('right_eye'), keypoints.index('right_ear')],
             [keypoints.index('left_eye'), keypoints.index('nose')],
             [keypoints.index('left_eye'), keypoints.index('left_ear')],
             [keypoints.index('right_shoulder'), keypoints.index('right_elbow')],
             [keypoints.index('right_elbow'), keypoints.index('right_wrist')],
             [keypoints.index('left_shoulder'), keypoints.index('left_elbow')],
             [keypoints.index('left_elbow'), keypoints.index('left_wrist')],
             [keypoints.index('right_hip'), keypoints.index('right_knee')],
             [keypoints.index('right_knee'), keypoints.index('right_ankle')],
             [keypoints.index('left_hip'), keypoints.index('left_knee')],
             [keypoints.index('left_knee'), keypoints.index('left_ankle')],
             [keypoints.index('right_shoulder'), keypoints.index('left_shoulder')],
             [keypoints.index('right_hip'), keypoints.index('left_hip')],
             [keypoints.index('right_shoulder'), keypoints.index('right_hip')],
             [keypoints.index('left_shoulder'), keypoints.index('left_hip')]]
    return limbs

def draw_skeleton_per_person(img, param, limbs, all_keypoints, all_scores, confs,
                             keypoint_threshold=2, conf_threshold=0.9):
    # Example:skeletal_img = draw_skeleton_per_person(img, output["keypoints"],
    # output["keypoints_scores"], output["scores"],keypoint_threshold=2)
    cmap = plt.get_cmap('rainbow')
    img_copy = img.copy()
    if len(param["keypoints"])>0:
        colors = np.arange(1, 255, 255 // len(all_keypoints)).tolist()[::-1]
        for person_id in range(len(all_keypoints)):
            if confs[person_id] > conf_threshold:
                keypoints = all_keypoints[person_id, ...]
                for limb_id in range(len(limbs)):
                    limb_loc1 = keypoints[limbs[limb_id][0], :2].detach().numpy().astype(np.int32)
                    limb_loc2 = keypoints[limbs[limb_id][1], :2].detach().numpy().astype(np.int32)
                    limb_score = min(all_scores[person_id, limbs[limb_id][0]], all_scores[person_id, limbs[limb_id][1]])
                    if limb_score > keypoint_threshold:
                        color = tuple(np.asarray(cmap(colors[person_id])[:-1]) * 255)
                        cv2.line(img_copy, tuple(limb_loc1), tuple(limb_loc2), color, 25)
    return img_copy
