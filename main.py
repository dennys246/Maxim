import argparse, os

from src.conscience.selfy import Maxim

def life():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--robot-name",
        default="reachy_mini",
        help="Reachy Mini daemon robot_name / zenoh namespace (default: $MAXIM_ROBOT_NAME or 'reachy_mini').",
    )
    parser.add_argument(
        "--home-dir",
        default=os.path.join("experiments","maxim"),
        help="Reachy Mini daemon robot_name / zenoh namespace (default: experiments/maxim').",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Seconds to wait for the Zenoh connection (default: 30).",
    )
    args = parser.parse_args()

    try:
        
        maxim = Maxim(robot_name=args.robot_name, timeout=args.timeout)

        print("✅ Maxim lives! ")

        while maxim.alive:
            maxim.live(home_dir = args.home_dir)

    except Exception as e:
        print("❌ Connection failed", e)
    
if __name__ == "__main__":

    life()
    
