import cv2, os, re
from glob import glob

def load_photos(photo_folder, count = 4):

    photo_files = glob(os.path.join(photo_folder, "*.png"))

    if count <= 0:
        return []

    indexed_photo_files = []
    for photo_file in photo_files:
        match = re.search(r"img_(\d+)\.png$", os.path.basename(photo_file))
        if match is None:
            continue
        indexed_photo_files.append((int(match.group(1)), photo_file))

    indexed_photo_files.sort(key=lambda item: item[0])
    selected_photo_files = indexed_photo_files[-count:]

    photos = []
    for _, photo_file in selected_photo_files:
        photo = cv2.imread(photo_file)
        if photo is None:
            continue
        photos.append(photo)

    return photos

def create_video(video_path, photos, fps):
    """ Stitch photos together into a video file. """
    # Create new video name
    output_path = f"{os.path.splitext(video_path)[0]}_reachy_memory.mp4"

    # Output the video to a .mp4
    height, width = photos[0].shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in photos:
        writer.write(frame)

    # Release the video to clear up memory
    writer.release()
