import math, time
from reachy_mini import ReachyMini
import numpy as np

from src.motion.movement import move_head

class Maxim:
    """
    Reachy-Mini modality threader for distributed computing
    """

    def __init__(self):
        self.alive = True

        self.name = "Maxim"
        self.start = time.time()
        self.reachy_ip = "192.168.50.149"

        self.mini = ReachyMini(localhost_only = False, spawn_daemon = True)

        self.x = 0
        self.y = 0
        self.z = 0

        self.roll = 0
        self.pitch = 0
        self.yaw = 0
    
    def live(self):
        for x in [-1, -2, -4, 4]:
            self.move(x = x)
        return

    def move(self, x = None, y = None, z = None, roll = None, pitch = None, yaw = None, duration = 1.0):
        for param, class_param in zip([x, y, z, roll, pitch, yaw], [self.x, self.y, self.z, self.roll, self.pitch, self.yaw]):
            if param and param != class_param:
                class_param = param

        move_head(self.mini, self.x, self.y, self.z, self.roll, self.pitch, self.yaw, self.duration)

    def hear(self):
        return

    def watch(self):
        return

    def learn(self):
        return

    def reflect(self):
        return
    
    def journal(self):
        return

    def sleep(self):
        return

    def thread(self, requests, nodes):
        # Assess computing capabilties between available connections
        # NOTE: Need to establish identity and connection protocol (443/80?)

        # Assess computing requirements between requests

        # Split modality requests between computing nodes

        # 
        return



if __name__ == "__main__":
    conscience = Maxim()
    