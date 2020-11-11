from .abstracts.ExtensionProcessor import ExtensionProcessor
import cv2
import numpy as np
from .Clip import Clip
import math
import skvideo.io

class VideoProcessor(ExtensionProcessor):
    def __init__(
        self, ext = '.avi', n_frames_per_video= 16, frame_dim=(16, 16, 1), 
        sliding_window=True, grayscale=True, debug=False
    ):
        self.ext = ext
        self.debug = debug
        self.n_frames_per_video = n_frames_per_video
        self.frame_dim = frame_dim
        self.sliding_window=sliding_window
        self.grayscale=grayscale
        
    def resize_image(self, image, dim):
        return cv2.resize(image, dim,interpolation=cv2.INTER_AREA)

    def count_frames(self, path):
        video = self.get_video(path)
        n_frames = np.shape(video)[0]
        
        if(self.debug == True):
            print(f"Shape of {path}: {np.shape(video)}")
            print(f"Frames of {path}: {n_frames}")
        return n_frames

    def get_video(self, video_path):
        return skvideo.io.vread(video_path, as_grey=self.grayscale, verbosity=0)

    def get_cropped_frames(self, video, start_frame=None):
        raw_frames = self.get_window_from_video(video, start_frame)

        if(self.debug == True):
            print(f"Shape of raw_frames: {np.shape(raw_frames)}")
            print(f"Shape of raw_frames[0]: {np.shape(raw_frames[0])}")

        frames = []
        for frame in raw_frames:
            if(self.grayscale == True):
                img = self.resize_image(frame, (self.frame_dim[0], self.frame_dim[1]))
  
        if(self.debug == True):
            print(f"Shape of frames: {np.shape(frames)}")
            print(f"Shape of frames[0]: {np.shape(frames[0])}")

        return np.array(frames)

    def get_window_from_video(self, video, start_frame=None):
        try:
            if(start_frame is None):
                return video[0:self.n_frames_per_video]

            return video[start_frame:start_frame + self.n_frames_per_video]
        except Exception as e:
            print(f"[{self.name}] - get_window_from_video ERROR")
            print(e)

    def get_clips(self, path, label):
        
        if(self.sliding_window == False):
            return Clip(path, label, 0, self.n_frames_per_video)
        
        n_frames = self.count_frames(path)
        
        clips = []
        n_windows = math.floor(n_frames / self.n_frames_per_video) - 1
        
        for i in range(n_windows):
            
            start = i * self.n_frames_per_video
            stop = (i + 1) * self.n_frames_per_video

            clip = Clip(path, 
                        label, 
                        start_frame= start, 
                        stop_frame= stop
                       )
            clips.append(clip)

        # if(self.debug == True):
        #     print(f"Number of frames : {n_frames}")
        #     print(f"Number of windows : {n_windows}")
        return clips