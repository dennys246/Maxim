import time
from typing import Optional

from reachy_mini import ReachyMini

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
        )

        self.x = 0.0
        self.y = 0.0
        self.z = 0.0

        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
    
    def live(self):
        for x in (-20, 0, 20, 0):
            self.move(x=x, duration=0.8)

    def move(
        self,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
        roll: Optional[float] = None,
        pitch: Optional[float] = None,
        yaw: Optional[float] = None,
        duration: Optional[float] = None,
    ) -> None:
        if duration is not None:
            self.duration = duration

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

    def hear(self):
        return

    def watch(self):
        return

    def learn(self):
        return

    def reflect(self):
        return
    
    def journal(self):
        return

    def sleep(self):
        return

    def thread(self, requests, nodes):
        # Assess computing capabilties between available connections
        # NOTE: Need to establish identity and connection protocol (443/80?)

        # Assess computing requirements between requests

        # Split modality requests between computing nodes

        # 
        return



if __name__ == "__main__":
    conscience = Maxim()
    
