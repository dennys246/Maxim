from src.motion.move import head_pose_matrix


def move_head(mini, x, y, z, roll, pitch, yaw, duration):
    pose = head_pose_matrix(x, y, z, roll, pitch, yaw)

    mini.set_target_head_pose(pose)

    mini.goto_target(duration = duration)
    
    print("Moved to position.")