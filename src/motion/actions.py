from src.motion.movement import move_head


class actionables:

    def __init__(self):
        self.actions = {
            "move": self.move,
        }


def sassy_no(maxim):

    maxim.move(yaw=30, duration=0.5)

def confused(maxim):
    
    maxim.move(roll=15, duration=0.5)

def head_bobble(maxim):

    maxim.move(pitch=20, duration=0.3)
    maxim.move(pitch=-20, duration=0.3)
    maxim.move(pitch=0, duration=0.3)