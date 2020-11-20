from .abstracts.ExtensionProcessor import ExtensionProcessor
import cv2
import numpy as np
from .Clip import Clip
import math
import sys
import logging

logger = logging.getLogger(__name__)

class NpyProcessor(ExtensionProcessor):
    def __init__(
        self, ext = '.npy', n_frames_per_video= 16, frame_dim=None, 
        sliding_window=True, grayscale=True, debug=False
    ):
        self.ext = ext
        self.debug = debug
        self.n_frames_per_video = n_frames_per_video
        self.frame_dim = frame_dim
        self.sliding_window=sliding_window
        self.grayscale=grayscale
        self.name = 'NpyProcessor'
    
        self.max = 0
        self.mean = 0
        self.n_videos = 0

    def image_rgb_to_grayscale(self, image):
        logger.debug("Converting RGB image to Grayscale")
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return np.expand_dims(img, axis=-1)
        
    def resize_image(self, image, dim):
        logger.debug("Resizing image")

        if(np.shape(image) == dim):
            return image

        resized =  cv2.resize(image, (dim[0], dim[1]),interpolation=cv2.INTER_AREA)
            
        return resized
    def get_video(self, video_path):
        logger.debug(f"Getting video {video_path}")

        try:
            return np.load(video_path, allow_pickle=True)
        except Exception as e:
            print(f"[{self.name}] - get_video ERROR")
            print(e)
            
    def count_frames(self, path):
        logger.debug(f"Counting frames of video {path}")
        video = self.get_video(path)
        
        self.max += np.max(video)
        self.mean += np.mean(video)
        self.n_videos += 1
        
        n_frames = np.shape(video)[0]
        
        if(self.debug == True):
            print(f"Shape of {path}: {np.shape(video)}")
            print(f"Frames of {path}: {n_frames}")
        return n_frames
   
    def get_window_from_video(self, video, start_frame=None):
        logger.debug(f"Getting windows of video")
        try:
            if(start_frame is None):
                return video[0:self.n_frames_per_video]

            return video[start_frame:start_frame + self.n_frames_per_video]
        except Exception as e:
            print(f"[{self.name}] - get_window_from_video ERROR")
            print(e)
            
    def get_cropped_frames(self, video, start_frame=None):
        logger.debug(f"Getting cropped frames of video")

        try:
            raw_frames = self.get_window_from_video(video, start_frame)

            if(self.debug == True):
                print(f"Shape of video: {np.shape(video)}")
                print(f"Shape of raw_frames: {np.shape(raw_frames)}")
                print(f"Shape of raw_frames[0]: {np.shape(raw_frames[0])}")


            frames = []
            for frame in raw_frames:
                if(self.frame_dim is not None):
                    frame = self.resize_image(frame, self.frame_dim)

                    if(len(np.shape(frame)) > 2):
                        if(self.frame_dim[2] == 1 and np.shape(frame)[2] == 3):
                            frame = self.image_rgb_to_grayscale(frame)
                
                frames.append(frame)
                
            if(self.debug == True):
                print(f"Shape of frames: {np.shape(frames)}")
                print(f"Shape of frames[0]: {np.shape(frames[0])}")
                
            return np.array(frames)

        except Exception as e:
            print(f"[{self.name}] - get_cropped_frames ERROR")
            print(e)

    def get_clips(self, path, label):
        logger.debug(f"Getting clips of video {path}")

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

        if(self.debug == True):
            print(f"Number of frames : {n_frames}")
            print(f"Number of windows : {n_windows}")
        return clips