from reachy_mini import ReachyMini

class Maxim:

    def __init__(self):

        self.name = "Maxim"

        with ReachyMini() as self.mini:

            return self



    def move(self, x, y, z, roll, degrees, mm, duration):

        self.mini.goto_target( 
            head= ReachyMini.utils.create_head_pose(x,
                                                    y,
                                                    z, 
                                                    roll, 
                                                    degrees, 
                                                    mm),
                                                    duration = 1)
        return

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
    