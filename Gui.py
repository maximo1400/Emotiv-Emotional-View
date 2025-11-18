import tkinter as tk
import time
import os
import queue
from PIL import Image, ImageTk
import pickle


class EmotionsAppGui:
    def __init__(
        self,
        queue: queue,
        emotiv_profile=None,
        demo=False,
    ) -> None:
        self.time_img_s = 6
        self.time_after_img_s = 2  # Happens once before first image and then twice after each image
        self.canvas_width = 800
        self.canvas_height = 800
        self.winwdow_width = self.canvas_width + 25
        self.window_height = self.canvas_height + 100
        self.fig_margin = 10
        self.fig_size = 800
        self.emotiv_profile = emotiv_profile
        self.demo = demo
        self.queue = queue
        self.pictures_folder = "img"
        self.images = {}
        self.bg_color = "#808080"
        self.pickle_file = "OASIS_database_2016/all_info.pkl"

    def start(self) -> None:
        self.root = tk.Tk()
        self.label = tk.Label(font=("Helvetica", 18))
        self.root.configure(bg=self.bg_color)

        # self.label = tk.Label(font=("Helvetica", 18), bg=self.bg_color, fg="white")
        self.root.geometry(f"{self.winwdow_width}x{self.window_height}")
        # self.root.attributes("-fullscreen", True)
        self.load_images(self.pictures_folder)

        if not self.demo:
            self.root.attributes("-fullscreen", True)

        if self.emotiv_profile is None:

            self.get_profile_name()
            self.root.mainloop()
        else:
            self.canvas = tk.Canvas(
                self.root,
                width=self.canvas_width,
                height=self.canvas_height,
                bg=self.bg_color,
                bd=0,
                highlightthickness=0,
            )
            # self.canvas.pack(fill="both", expand=True, padx=10, pady=10)
            self.canvas.pack(padx=10, pady=10, side="bottom")
            self.wait_to_start()

    def get_profile_name(self) -> None:
        self.label.config(text=f"ingrese nombre perfil")
        self.label.pack(padx=10, pady=10)

        self.textobox = tk.Text(self.root, height=1, width=10, font=("Helvetica", 16))
        self.textobox.bind("<KeyPress>", self.enter_on_get_profile_name)
        self.textobox.pack(padx=10, pady=10)

        self.button = tk.Button(self.root, text="Aceptar", command=self.close_profile_window)
        self.button.pack(padx=10, pady=10)

    def enter_on_get_profile_name(self, event) -> None:
        # print(event)
        if event.keysym == "Return" and event.keycode == 13:
            self.close_profile_window()

    def close_profile_window(self) -> None:
        self.emotiv_profile = self.textobox.get("1.0", "end-1c")
        # print(self.emotiv_profile)

        # Hide button and textbox
        self.button.pack_forget()
        self.textobox.pack_forget()

        self.root.destroy()

    def wait_to_start(self) -> None:
        msg = f"""A continuacion se presentan imagens

        Cuando estés listo, presiona Enter para iniciar."""
        self.canvas.create_text(
            self.canvas_width / 2,
            self.canvas_height / 2,
            text=msg,
            font=("Helvetica", 18),
            fill="black",
            justify="center",
            width=500,
        )
        self.canvas.update()
        self.root.bind("<KeyPress>", self.return_shortcut)
        self.root.mainloop()

    def return_shortcut(self, event) -> None:
        # print(event)
        if event.keysym == "Return" and event.keycode == 13:
            self.canvas.delete("all")
            self.canvas.update()
            self.root.unbind("<KeyPress>")
            time.sleep(0.1)
            if not self.demo:
                self.image_loop()
            else:
                self.demo_trainning_loop()

    def display_text_on_trainning(self, text) -> None:
        # self.label.config(text=f"Perfil: {self.emotiv_profile} \n Comando siendo entrenado: {text}")
        self.label.config(text=f"Imagen: {text}")
        self.label.pack(padx=10, pady=10)

    def image_loop(self) -> None:
        order = []
        with open(self.pickle_file, "rb") as f:
            obj = pickle.load(f)
            order = obj["img_order"]
        # print(f"Image order loaded from pickle: {order}")
        complete_loop = []
        images = self.images.keys()
        images = list(images)

        # for i in range(len(images)):
        #     complete_loop += [images[i], "next_round"]

        # complete_loop[-1] = "end"
        complete_loop = order + ["end"]
        print(f"lista comandos: {complete_loop}")

        for img in complete_loop:
            self.await_and_update_queue(img)
            self.show_image(img)
        self.label.config(text=f"Gracias!", font=("Helvetica", 32))
        self.label.pack(padx=10, pady=10)
        self.canvas.update()
        time.sleep(4)
        self.root.destroy()

    def await_and_update_queue(self, img) -> None:
        time.sleep(self.time_after_img_s)
        while not self.queue.empty():
            time.sleep(0.1)

        self.queue.put(img)
        print(f"Image <<'{img}'>> sent to queue")

        while not self.queue.empty():
            time.sleep(0.1)

    def load_images(self, folder=None) -> None:
        """Load supported images from folder into self.images as ImageTk.PhotoImage.
        Keys are filenames without extensions."""
        if folder is None:
            folder = self.pictures_folder

        self.images = {}
        if not os.path.isdir(folder):
            return
        supported = (".png", ".jpg", ".jpeg", ".gif", ".bmp")
        for fname in os.listdir(folder):
            if not fname.lower().endswith(supported):
                continue
            path = os.path.join(folder, fname)
            try:
                img = Image.open(path)
                # optional resize to fig_size (keeps aspect ratio)
                if self.fig_size:
                    resample = getattr(Image, "Resampling", Image).LANCZOS
                    img = img.resize((self.fig_size, self.fig_size), resample)
                tkimg = ImageTk.PhotoImage(img)
                key = os.path.splitext(fname)[0]
                self.images[key] = tkimg
            except Exception as e:
                print(f"Failed to load image {path}: {e}")

    def show_image(self, image_key) -> None:
        print(f"Showing image: {image_key}")
        time_s = self.time_img_s
        t_end_p = time.time() + time_s
        t_end_w = t_end_p + self.time_after_img_s
        self.canvas.delete("all")
        if image_key in self.images:
            img = self.images[image_key]
            x = (self.canvas_width - img.width()) // 2
            y = (self.canvas_height - img.height()) // 2
            self.canvas.create_image(x, y, anchor="nw", image=img)
            while time.time() < t_end_p:
                self.canvas.update()
            print(f"Image {image_key} display time ended, starting rest period")
            self.canvas.delete("all")
            while time.time() < t_end_w:
                self.canvas.update()
            print(f"Image {image_key} time ended")
        else:
            print(f"No image found for key: {image_key}")
