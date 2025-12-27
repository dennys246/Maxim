from src.conscience.selfy import Maxim

def live():
    
    try:
        life()
        print("✅ Reachy Mini connected (simulation running)")
    except Exception as e:
        print("❌ Connection failed:", e)

def life():
    maxim = Maxim()

    maxim.move(
        x=0.0,
        y=0.0,
        z=1.0,
        roll=0.0,
        pitch=0.0,
        yaw=0.0,
        duration=2.0
    )

if __name__ == "__main__":
    live()
    