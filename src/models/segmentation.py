import random, cv2, os, json, os
import re
from glob import glob
from ultralytics import YOLO

class YOLO8:
    def __init__(self, pose_model = False):
        # Initialize models
        self.model = YOLO("yolov8m-seg.pt") 

        # Define confidence thresholds
        self.conf = 0.5

        if pose_model:
            # Load YOLO pose model
            self.pose_model = YOLO("yolov8m-pose.pt")

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
        frame_ind = 0
        for photo in photos[:]:
            # Track people in this frame
            results = self.model.track(
                photo,
                classes=[0, 1, 2, 3, 4], 
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
                
                # If we have a bounding box found
                if box.id:
                    # format the info and add to list of found people
                    track_id = int(box.id)
                    x1, y1, x2, y2 = box.xyxy.cpu().numpy().squeeze()
                    conf = float(box.conf)

                    # Preserve observation
                    observations.append([track_id, frame_ind, x1, y1, x2, y2, conf])
            

        if observations: # If person detected
            print(f"Maxim found something in their view!")
            return observations
        else:
            print("Nothing interesting found.")

