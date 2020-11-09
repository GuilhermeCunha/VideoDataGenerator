class Clip:
    def __init__(self, video_path, label, start_frame, stop_frame):
        self.video_path = video_path
        self.start_frame = start_frame
        self.stop_frame = stop_frame
        self.label = label