import os

def build_home(home_dir):
    if os.path.exists(os.path.join(home_dir, "images")):
        print("Home already built")
        return

    os.makedirs(os.path.join(home_dir, "images"), exist_ok = True)
    os.makedirs(os.path.join(home_dir, "videos"), exist_ok = True)
    os.makedirs(os.path.join(home_dir, "audio"), exist_ok = True)
    os.makedirs(os.path.join(home_dir, "text"), exist_ok = True)
    os.makedirs(os.path.join(home_dir, "logs"), exist_ok = True)
    return