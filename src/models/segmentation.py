import random, cv2, os, json, os
import re
from glob import glob
import numpy as np
from ultralytics import YOLO

class YOLO8:
    def __init__(self, pose_model = False):
        # Initialize models
        self.model = YOLO("experiments/models/yolo8m-seg.pt") 

        # Define confidence thresholds
        self.conf = 0.5

        if pose_model:
            # Load YOLO pose model
            self.pose_model = YOLO("experiments/models/yolo8m-pose.pt")

            self.keypoint_conf = 0.25

            # COCO keypoints order used
            self.coco_keypoints = [
                "nose",
                "left_eye",
                "right_eye",
                "left_ear",
                "right_ear",
                "left_shoulder",
                "right_shoulder",
                "left_elbow",
                "right_elbow",
                "left_wrist",
                "right_wrist",
                "left_hip",
                "right_hip",
                "left_knee",
                "right_knee",
                "left_ankle",
                "right_ankle",
            ]


    def segment_photos(self, photos, interests = [0, 1, 2, 3, 4], display = False, save_video = False):

        observations = [] # Things of interest
        if photos is None:
            return observations

        if isinstance(photos, np.ndarray):
            photos = [photos]

        for frame_ind, photo in enumerate(photos):
            if isinstance(photo, np.ndarray):
                if photo.ndim == 2:
                    photo = cv2.cvtColor(photo, cv2.COLOR_GRAY2BGR)
                elif photo.ndim == 3 and photo.shape[2] == 1:
                    photo = cv2.cvtColor(photo, cv2.COLOR_GRAY2BGR)
                elif photo.ndim == 3 and photo.shape[2] == 4:
                    photo = cv2.cvtColor(photo, cv2.COLOR_BGRA2BGR)

            if not (isinstance(photo, np.ndarray) and photo.ndim == 3 and photo.shape[2] == 3):
                continue

            # Track people in this frame
            results = self.model.track(
                photo,
                classes=interests,
                conf=0.5,
                persist=True 
            )

            # Attempt to detect poses 
            #pose_results = self.pose_model(
            #    photo,
            #    classes=[0], 
            #    conf=self.conf
            #)

            for box in results[0].boxes:
                
                track_id = int(box.id) if box.id is not None else None
                x1, y1, x2, y2 = box.xyxy.cpu().numpy().squeeze()
                conf = float(box.conf)

                # Preserve observation
                observations.append([track_id, frame_ind, x1, y1, x2, y2, conf])
            

        if observations: # If person detected
            print(f"Maxim found something in their view!")
            return observations
        else:
            print("Nothing interesting found.")
