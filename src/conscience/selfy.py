import os

#os.environ["REACHY_MEDIA_BACKEND"] = "zenoh"
#os.environ["REACHY_DISABLE_WEBRTC"] = "1"
#os.environ["GST_DISABLE_REGISTRY_FORK"] = "1"

import json, random
import time, atexit, cv2
from typing import Optional
from scipy.signal import resample
from reachy_mini import ReachyMini
from scipy.io.wavfile import write

from src.motion.movement import move_head, load_actions
from src.utils.data_management import build_home

from src.video.sight import load_photos, create_video
#from src.models.segmentation import YOLO8

os.environ["PYOPENGL_PLATFORM"] = "egl"

class Maxim:
    """
    Reachy-Mini modality threader for distributed computing
    """

    def __init__(
        self,
        robot_name: str = "reachy_mini",
        timeout: float = 30.0,
        media_backend: str = "no_media",  # avoid WebRTC/GStreamer if signalling is down
    ):
        self.alive = True

        self.name = robot_name or os.getenv("MAXIM_ROBOT_NAME", "reachy_mini")
        self.start = time.time()
        self.duration = 1.0

        self.epoch = 0

        self.observation_period = 5

        self.actions = load_actions()

        # robot_name must match the daemon namespace (default: reachy_mini).
        # localhost_only=False enables zenoh peer discovery across the LAN.
        self.mini = ReachyMini(
            robot_name=self.name,
            localhost_only=False,
            spawn_daemon=False,
            use_sim=False,
            timeout=timeout,
            media_backend=media_backend,
        )

        self.x = 0.01
        self.y = 0.01
        self.z = 0.01

        self.roll = 0.01
        self.pitch = 0.01
        self.yaw = 0.01

        atexit.register(self.sleep)
    
    def live(self, home_dir):

        self.home_dir = home_dir
        build_home(home_dir)

        while True:
            self.epoch += 1

            self.look(os.path.join(home_dir, "memories", "images", f"img_{self.epoch}.png"))

            #self.hear(os.path.join(home_dir, "memories","audio", f"reachy_audio_{self.epoch}.mp3"))

            if self.observation_period and self.epoch % self.observation_period == 0:

                self.observe()

            if  self.epoch > self.duration:
                break


        self.sleep()

    def move(
        self,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
        roll: Optional[float] = None,
        pitch: Optional[float] = None,
        yaw: Optional[float] = None,
        duration: Optional[float] = None) -> None:

        """
        Docstring for move
        
        :param self: Description
        :param x: Description
        :type x: Optional[float]
        :param y: Description
        :type y: Optional[float]
        :param z: Description
        :type z: Optional[float]
        :param roll: Description
        :type roll: Optional[float]
        :param pitch: Description
        :type pitch: Optional[float]
        :param yaw: Description
        :type yaw: Optional[float]
        :param duration: Description
        :type duration: Optional[float]
        """ 
        
        # Update duration if specified
        if duration is not None:
            self.duration = duration


        # Update only specified parameters
        updates = {
            "x": x,
            "y": y,
            "z": z,
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
        }
        for attr, val in updates.items():
            if val is not None:
                setattr(self, attr, val)

        # Execute head movement
        move_head(
            self.mini,
            self.x,
            self.y,
            self.z,
            self.roll,
            self.pitch,
            self.yaw,
            self.duration,
        )

    def act(self, action):
        for movement in self.actions[action]["movements"]:
            self.move(
                movement[0],
                movement[1],
                movement[2],
                movement[3],
                movement[4],
                movement[5],
                movement[6]
            )
            time.sleep(movement[6])

    def hear(self, save_file = None):
        # Grab audio samples from reachy mini microphone
        try:
            samples = self.mini.media.get_audio_sample()
        except Exception as e:
            print(f"[WARN] Failed to capture audio sample: {e}")
            return None

        if samples is None or len(samples) == 0:
            print("[WARN] Empty audio sample received.")
            return None

        # Resample to local rate
        num_samples = int(
            self.mini.media.get_output_audio_samplerate()
            * len(samples)
            / self.mini.media.get_input_audio_samplerate()
        )
        samples = resample(samples, num_samples)
        
        if save_file:
            # Save audio samples to file
            os.makedirs(os.path.dirname(save_file) or ".", exist_ok=True)
            try:
                write(save_file, self.mini.media.get_output_audio_samplerate(), samples)
            except Exception as e:
                print(f"[WARN] Failed to write audio to '{save_file}': {e}")
        # Return audio samples
        return samples
    
    def speak(self, samples):
        # Push audio samples to reachy mini speaker
        self.mini.media.push_audio_sample(samples)
        return

    def look(self, save_file = None):
        # Grab frame from reachy mini camera
        try:
            frame = self.mini.media.get_frame()
        except Exception as e:
            print(f"[WARN] Failed to capture frame: {e}")
            return None

        is_empty = frame is None
        if not is_empty and hasattr(frame, "size"):
            is_empty = frame.size == 0
        if is_empty:
            print("[WARN] Empty frame received.")
            return None
        
        # Save frame to file if specified
        if save_file is not None:
            os.makedirs(os.path.dirname(save_file) or ".", exist_ok=True)
            try:
                ok = cv2.imwrite(save_file, frame)
                if not ok:
                    print(f"[WARN] Failed to write image to '{save_file}'.")
            except Exception as e:
                print(f"[WARN] Failed to write image to '{save_file}': {e}")
        return frame

    def learn(self):
        return

    def observe(self, epochs = 10):
        
        # Grab last epochs
        home_dir = getattr(self, "home_dir", None)
        if not home_dir:
            print("[WARN] No home directory set; skipping observation.")
            return

        photos = load_photos(os.path.join(home_dir, "memories", "images"), count = epochs)
        if not photos:
            print("[WARN] No photos available for observation.")
            return

        photo_shape = photos[0].shape
        photo_height, photo_width = photo_shape[0], photo_shape[1]
        photo_center = [photo_width / 2, photo_height / 2]

        # Segment photos
        observations = self.segmenter.segment_photos(photos)
        if not observations:
            return

        # Grab a random observation
        observation = random.choice(observations)

        # Calculate movement to center segmentation
        observation_center = [(observation[2] + observation[4])/2, (observation[3] + observation[5])/2]

        x_diff = (observation_center[0] - photo_center[0])
        y_diff = (observation_center[1] - photo_center[1])

        pitch_estimate = (y_diff / photo_height) * 10
        yaw_estimate = (x_diff / photo_width) * 10

        # Create random roll
        random_roll = random.randint(-5, 5)

        # Initiate movement
        self.move(roll = random_roll, pitch = pitch_estimate, yaw = yaw_estimate)

        return
    
    def journal(self):
        entry = {
            "date": time.time(),
            "epoch": self.epoch,
        }

        json.loads("")
        return
    
    def awaken(self, vision = True, audio = True):
        # Load models
        if vision:
            self.segmenter = YOLO8() # Visual segmentation model

        # Wake up Reachy
        self.mini.wake_up()

        return

    def sleep(self):
        # Send Reachy to sleep
        self.mini.goto_sleep()

        # Clear memroy of models not being trained
        del self.segmenter

        return

    def thread(self, requests, nodes):
        # Assess computing capabilties between available connections
        # NOTE: Need to establish identity and connection protocol (443/80?)

        # Assess computing requirements between requests

        # Split modality requests between computing nodes

        return



if __name__ == "__main__":
    conscience = Maxim()
    
