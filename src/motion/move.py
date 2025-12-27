import math
import numpy as np

def head_pose_matrix(x=0, y=0, z=0, roll=0, pitch=0, yaw=0):
    # Convert units
    x, y, z = x / 1000, y / 1000, z / 1000
    roll, pitch, yaw = map(math.radians, (roll, pitch, yaw))

    # Rotation matrices
    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(roll), -math.sin(roll)],
        [0, math.sin(roll),  math.cos(roll)],
    ])

    Ry = np.array([
        [ math.cos(pitch), 0, math.sin(pitch)],
        [ 0,               1, 0],
        [-math.sin(pitch), 0, math.cos(pitch)],
    ])

    Rz = np.array([
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw),  math.cos(yaw), 0],
        [0,              0,             1],
    ])

    R = Rz @ Ry @ Rx

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]

    return T