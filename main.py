from src.conscience.selfy import Maxim

def life():
    try:
        
        maxim = Maxim()

        print("✅ Maxim lives! ")

        while maxim.alive:
            maxim.live()

    except Exception as e:
        print("❌ Connection failed", e)
    
if __name__ == "__main__":

    life()
    