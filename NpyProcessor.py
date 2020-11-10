from .abstracts.ExtensionProcessor import ExtensionProcessor
import cv2
import numpy as np
from .Clip import Clip
import math

class NpyProcessor(ExtensionProcessor):
    def __init__(
        self, ext = '.npy', n_frames_per_video= 16, frame_dim=(16, 16, 1), 
        sliding_window=True, grayscale=True, debug=False
    ):
        self.ext = ext
        self.debug = debug
        self.n_frames_per_video = n_frames_per_video
        self.frame_dim = frame_dim
        self.sliding_window=sliding_window
        self.grayscale=grayscale


    def image_rgb_to_grayscale(self, image):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return np.expand_dims(img, axis=-1)
        
    def resize_image(self, image, dim):
        return cv2.resize(image, dim,interpolation=cv2.INTER_AREA)

    def get_video(self, video_path):
        return np.load(video_path, allow_pickle=True)
    
    def count_frames(self, path):
        return len(self.get_video(path))
   
    def get_cropped_frames(self, video, start_frame=None):
        raw_frames = self.get_window_from_video(video, start_frame)

        if(self.debug == True):
            print(f"Shape of raw_frames: {np.shape(raw_frames)}")
            print(f"Shape of raw_frames[0]: {np.shape(raw_frames[0])}")


        frames = []
        for frame in raw_frames:
            if(self.grayscale == True):
                img = self.resize_image(frame, (self.frame_dim[0], self.frame_dim[1]))
                img = self.image_rgb_to_grayscale(img)
  
        if(self.debug == True):
            print(f"Shape of frames: {np.shape(frames)}")
            print(f"Shape of frames[0]: {np.shape(frames[0])}")
        return np.array(frames)


    def get_window_from_video(self, video, start_frame=None):
        if(start_frame is None):
            return video[:self.n_frames_per_video]
        
        return video[start_frame:start_frame + self.n_frames_per_video]

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