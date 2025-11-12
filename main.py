import time
import threading
import os
import queue
from dotenv import load_dotenv

load_dotenv()

from Emotiv import Train
from Gui import EmotionsAppGui

your_app_client_id = os.getenv("APP_CLIENT_ID")
your_app_client_secret = os.getenv("APP_CLIENT_SECRET")

def main() -> None:
    # Please fill your application clientId and clientSecret before running script

    # Init Train
    is_demo = False
    profile_name = "virtual"
    img_names_queue = queue.Queue()

    # list data streams
    streams = ["mot", "dev", "eq", "pow", "met", "com", "fac", "sys"]
    streams = ["mot", "dev", "pow", "met", "com", "sys"]
    

    if profile_name is None:
        print("No profile name provided")
        return

    gui = EmotionsAppGui(
        img_names_queue,
        emotiv_profile=profile_name,
        demo=is_demo,
    )
    emotiv = Train(
        your_app_client_id,
        your_app_client_secret,
        verbose=False,
        emotiv_profile=profile_name,
    )

    t0 = time.time()
    t = threading.Thread(target=emotiv.start, args=[profile_name, streams, img_names_queue])
    # t = threading.Thread(target=emotiv.start, args=(streams,))
    t.start()
    gui.start()

    t.join()
    emotiv.unsub(streams)
    t1 = time.time()
    print(f"Tiempo de Aplicacion: {(t1-t0)/60.}min")

    if os.path.exists(f"sub_data/{profile_name}"):
        profile_name = profile_name + "_" + str(int(time.time()))

    os.mkdir(f"sub_data/{profile_name}")
    for stream in emotiv.data:
        emotiv.data[stream].to_csv(f"sub_data/{profile_name}/{stream}.csv")


if __name__ == "__main__":
    main()
