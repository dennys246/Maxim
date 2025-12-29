import os

def build_home(home_dir):
    if os.path.exists(os.path.join(home_dir, "memories", "images")):
        print("Home already built")
        return

    os.makedirs(os.path.join(home_dir, "memories", "images"), exist_ok = True)
    os.makedirs(os.path.join(home_dir, "memories", "videos"), exist_ok = True)
    os.makedirs(os.path.join(home_dir, "memories", "audio"), exist_ok = True)
    os.makedirs(os.path.join(home_dir, "memories", "text"), exist_ok = True)
    os.makedirs(os.path.join(home_dir, "logs"), exist_ok = True)
    return