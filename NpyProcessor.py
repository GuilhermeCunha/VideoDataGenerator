from .abstracts.ExtensionProcessor import ExtensionProcessor
import cv2
import numpy as np
from .Clip import Clip
from .utils import windows

import math
import sys
import logging


logger = logging.getLogger(__name__)

class NpyProcessor(ExtensionProcessor):
    def __init__(
        self, ext = '.npy', n_frames_per_video= 16, frame_dim=None,
        windowing= None, debug=False
    ):
        if(frame_dim is None):
            raise ValueError("'frame_dim' needs to be inserted. Ex.: (N_ROWS, N_COLS, N_CHANNELS)")
        
        if(len(frame_dim) != 3 ):
            raise ValueError("'frame_dim' must have size 3. Ex.: (N_ROWS, N_COLS, N_CHANNELS)")
        
        if(ext != '.npy'):
            raise ValueError("'ext' invalid. must be '.npy'")

        if(n_frames_per_video < 2):
            raise ValueError("'n_frames_per_video' must be greater than 1")

        if(windowing in [None, 'normal', 'sliding'] == False):
            raise ValueError("'windowing' is invalid. Must be in [None, 'normal', 'sliding']")

        self.ext = ext
        self.debug = debug
        self.n_frames_per_video = n_frames_per_video
        self.frame_dim = frame_dim
        self.name = 'NpyProcessor'
        self.windowing = windowing
        self.max = 0
        self.mean = 0
        self.min = 0

        self.n_videos = 0

    def image_rgb_to_grayscale(self, image):
        logger.debug("Converting RGB image to Grayscale")
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return np.expand_dims(img, axis=-1)
        

    def need_resize(self, image):
        if(len(np.shape(image)) > 2):
            return np.shape(image) != np.shape(self.frame_dim)
        
        return np.shape(image) != (np.shape(self.frame_dim)[0], np.shape(self.frame_dim)[1])

    def resize_image(self, image, dim):
        logger.debug("Resizing image")

        if(self.need_resize(image) is False):
            return image

        resized =  cv2.resize(image, (dim[0], dim[1]),interpolation=cv2.INTER_AREA)


        if(len(np.shape(resized)) <= 2):
            resized = np.expand_dims(resized, axis=-1)

        return resized

    def get_video(self, video_path):
        logger.debug(f"Getting video {video_path}")

        try:
            video = np.load(video_path, allow_pickle=True)

            self.max += np.max(video)
            self.mean += np.mean(video)
            self.min += np.min(video)

            self.n_videos += 1

            return video
        except Exception as e:
            print(f"[{self.name}] - get_video ERROR")
            print(e)
    
    def get_mean(self):
        return self.mean / self.n_videos

    def get_max(self):
        return self.max / self.n_videos
          
    def count_frames(self, path):
        logger.debug(f"Counting frames of video {path}")
        video = self.get_video(path)    
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
                frame = self.resize_image(frame, self.frame_dim)

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
        n_frames = self.count_frames(path)
        clips = []
    
        if self.windowing is not None:
            if(self.windowing == 'sliding'):
                windows = windows.get_sliding_windows(n_frames, self.n_frames_per_video)
            
            if(self.windowing == 'normal'):
                windows = windows.get_windows(n_frames, self.n_frames_per_video)

            for start, stop in windows:
                clip = Clip(path, 
                                label, 
                                start_frame= start, 
                                stop_frame= stop
                            )
                clips.append(clip)

            return clips

        clips.append(Clip(path, label, 0, self.n_frames_per_video))

        return clips