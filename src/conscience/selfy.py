import time, atexit, cv2
from typing import Optional
from scipy.signal import resample
from reachy_mini import ReachyMini
from scipy.io.wavfile import write

from src.motion.movement import move_head

class Maxim:
    """
    Reachy-Mini modality threader for distributed computing
    """

    def __init__(
        self,
        reachy_ip: str = "192.168.50.149",
        robot_name: str = "reachy_mini",
        timeout: float = 30.0,
        #media_backend: str = "no_media",  # avoid WebRTC/GStreamer if signalling is down
    ):
        self.alive = True

        self.name = robot_name
        self.start = time.time()
        self._reachy_ip = reachy_ip
        self.duration = 1.0

        # robot_name must match the daemon namespace (default: reachy_mini).
        # localhost_only=False enables zenoh peer discovery across the LAN.
        self.mini = ReachyMini(
            robot_name=self.name,
            localhost_only=False,
            spawn_daemon=False,
            use_sim=False,
            timeout=timeout,
            #media_backend=media_backend,
        )

        self.awaken()

        self.x = 0.0
        self.y = 0.0
        self.z = 0.0

        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0

        atexit.register(self.sleep)
    
    def live(self):
        for x in range(20):
            self.move(x=x, duration=0.7)

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

    def hear(self, save_file = None):
        # Grab audio samples from reachy mini microphone
        samples = self.mini.media.get_audio_sample()

        # Resample to local rate
        samples = resample(samples, self.mini.media.get_output_audio_samplerate()*len(samples)/ self.mini.media.get_input_audio_samplerate())
        
        if save_file:
            # Save audio samples to file
            write(save_file, self.mini.media.get_output_audio_samplerate(), samples)
        # Return audio samples
        return samples
    
    def speak(self, samples):
        # Push audio samples to reachy mini speaker
        self.mini.media.push_audio_sample(samples)
        return

    def look(self, save_file = None):
        # Grab frame from reachy mini camera
        frame = self.mini.media.get_frame()
        
        # Save frame to file if specified
        if save_file is not None:
            cv2.imwrite(save_file, frame)
        return frame

    def learn(self):
        return

    def reflect(self):
        return
    
    def journal(self):
        return
    
    def awaken(self):
        self.mini.wake_up()
        return

    def sleep(self):
        self.mini.goto_sleep()
        return

    def thread(self, requests, nodes):
        # Assess computing capabilties between available connections
        # NOTE: Need to establish identity and connection protocol (443/80?)

        # Assess computing requirements between requests

        # Split modality requests between computing nodes

        return



if __name__ == "__main__":
    conscience = Maxim()
    
