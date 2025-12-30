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

from src.motion.movement import move_antenna, move_head, load_actions
from src.utils.data_management import build_home

from src.camera.sight import load_photos, create_video
from src.camera.display import show_photo
from src.conscience.observation import passive_observation
from src.models.segmentation import YOLO8

os.environ["PYOPENGL_PLATFORM"] = "egl"

class Maxim:
    """
    A class for orchestracting models and agents with Reachy-Mini's.
    """

    def __init__(
        self,
        robot_name: str = "reachy_mini",
        timeout: float = 30.0,
        media_backend: str = "default",  # avoid WebRTC/GStreamer if signalling is down
        home_dir: str = "experiments/maxim/",
        epochs: int = 1000
    ):
        self.alive = True
        self._closed = False

        self.name = robot_name or os.getenv("MAXIM_ROBOT_NAME", "reachy_mini")
        self.start = time.time()
        self.duration = 1.0
        self.home_dir = home_dir

        self.current_epoch = 0
        self.epochs = epochs

        self.observation_period = 5

        self.interests = [0, 1, 2, 3, 4, 5]

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

        self.mini.start_recording()

        self.x = 0.01
        self.y = 0.01
        self.z = 0.01

        self.roll = 0.01
        self.pitch = 0.01
        self.yaw = 0.01

        atexit.register(self.sleep)

    def _release_cv2(self) -> None:
        try:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        except Exception:
            pass

    def _release_media(self) -> None:
        mini = getattr(self, "mini", None)
        if mini is None:
            return

        try:
            mini.media.close()
        except Exception as e:
            print(f"[WARN] Failed to close media: {e}")

        self._release_cv2()
    
    def live(self, home_dir: Optional[str] = None):

        self.awaken()

        if home_dir is not None:
            self.home_dir = home_dir

        build_home(self.home_dir)

        try:
            while True:
                self.current_epoch += 1

                photos = self.look(os.path.join(self.home_dir, "images", f"img_{self.current_epoch}.png"), show = False)

                #self.hear(os.path.join(self.home_dir, "memories","audio", f"reachy_audio_{self.epoch}.mp3"))

                if self.observation_period and self.current_epoch% self.observation_period == 0:

                    passive_observation(self, photos)

                if  self.current_epoch > self.epochs:
                    break
        finally:
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

    def move_antenna(
        self,
        right: Optional[float] = None,
        left: Optional[float] = None,
        angle: Optional[float] = None,
        duration: Optional[float] = None,
        method: str = "minjerk",
        degrees: bool = True,
        relative: bool = False,
    ) -> None:
        if angle is not None:
            right = angle
            left = angle
        if duration is None:
            duration = self.duration

        move_antenna(
            self.mini,
            right=right,
            left=left,
            duration=duration,
            method=method,
            degrees=degrees,
            relative=relative,
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

    def look(self, save_file = None, show = True, release = False):
        # Grab frame from reachy mini camera
        frame = None
        try:
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
            
            # Show frame if requested
            if show:
                try:
                    show_photo(frame)
                except Exception as e:
                    print(f"[WARN] Failed to display frame: {e}")
                finally:
                    self._release_cv2()
            
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
        finally:
            if release:
                self._release_media()

    def learn(self):
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
        if getattr(self, "_closed", False):
            return
        self._closed = True

        # Send Reachy to sleep
        try:
            self.mini.goto_sleep()
        except Exception as e:
            print(f"[WARN] Failed to send Reachy to sleep: {e}")

        # Stop recording data
        try:
            self.mini.stop_recording()
        except Exception as e:
            print(f"[WARN] Failed to stop recording: {e}")

        # Release the camera + any OpenCV resources
        self._release_media()


        return

    def thread(self, requests, nodes):
        # Assess computing capabilties between available connections
        # NOTE: Need to establish identity and connection protocol (443/80?)

        # Assess computing requirements between requests

        # Split modality requests between computing nodes

        return



if __name__ == "__main__":
    conscience = Maxim()
    
