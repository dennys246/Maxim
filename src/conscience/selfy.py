import math
from reachy_mini import ReachyMini
import numpy as np

from src.motion.move import head_pose_matrix

class Maxim:

    def __init__(self):

        self.name = "Maxim"
        self.reachy_ip = "192.168.50.149"

        self.mini = ReachyMini()

    def move(self, x, y, z, roll, pitch, yaw, duration):
        pose = head_pose_matrix(x, y, z, roll, pitch, yaw)

        self.mini.set_target_head_pose(pose)

        self.mini.goto_target(duration=duration)
        print("Moved to position.")

    def listen():
        return

    def watch():
        return

    def feel():
        return

    def learn():
        return

    def test(self):
        return
    
    def release():
        return

    def sleep():
        return
    
    def thread():
        return


if __name__ == "__main__":
    conscience = Maxim()
    