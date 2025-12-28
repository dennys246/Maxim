from src.motion.move import head_pose_matrix


def move_head(mini, x, y, z, roll, pitch, yaw, duration):
    pose = head_pose_matrix(x, y, z, roll, pitch, yaw)
    mini.goto_target(head=pose, duration=duration, body_yaw=None)
