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
        help="Reachy Mini home directory to save audio/pictures, models, and derivatives(default: experiments/maxim').",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Seconds to wait for the Zenoh connection (default: 30).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="Epochs to run Maxim for.",
    )

    args = parser.parse_args()

    try:
        
        maxim = Maxim(robot_name=args.robot_name, home_dir=args.home_dir, timeout=args.timeout, epochs=args.epochs)

        print("✅ Maxim lives! ")

        maxim.live(home_dir = args.home_dir)

    except Exception as e:
        print("❌ Connection failed", e)
    
if __name__ == "__main__":

    life()
    
