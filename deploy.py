import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
from process import process_frame

class App:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        # Video source and capture setup
        self.video_source = 0  # Use the default camera
        self.vid = cv2.VideoCapture(self.video_source)
        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set width
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set height
        
        # Create a canvas for video stream
        self.canvas = tk.Canvas(window, width=640, height=480)
        self.canvas.pack()
        
        # State to control processing
        self.processing_enabled = False
        
        # Toggle button to start/stop gesture recognition
        self.btn_toggle = ttk.Button(window, text="Enable Processing", command=self.toggle_processing)
        self.btn_toggle.pack(anchor=tk.CENTER, expand=True)

        # Update & display frames in the Tkinter window
        self.update()

        self.window.mainloop()

    def toggle_processing(self):
        """Toggle the state of gesture processing."""
        self.processing_enabled = not self.processing_enabled
        # Update button text based on state
        self.btn_toggle.config(text="Disable Processing" if self.processing_enabled else "Enable Processing")

    def update(self):
        """Read and display the next frame from the video source."""
        ret, frame = self.vid.read()
        if ret:
            frame = cv2.flip(frame, 1)  # Add this right after reading the frame to mirror the image
            frame = cv2.resize(frame, (640, 480))  # Resize frame to fit the canvas
            if self.processing_enabled:
                frame = process_frame(frame)  # Process the frame if enabled
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.window.after(15, self.update)  # Refresh every 15 milliseconds

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

App(tk.Tk(), "Tkinter and OpenCV")