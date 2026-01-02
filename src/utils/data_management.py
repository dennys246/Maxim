import os

def build_home(home_dir):
    for name in ("images", "videos", "audio", os.path.join("audio", "chunks"), "text", "logs", "models"):
        os.makedirs(os.path.join(home_dir, name), exist_ok=True)
