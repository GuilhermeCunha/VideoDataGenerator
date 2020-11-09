from .abstracts.ExtensionProcessor import ExtensionProcessor
import cv2
import numpy as np
from .Clip import Clip
import math

class VideoProcessor(ExtensionProcessor):
    def __init__(
        self, ext = '.mp4', n_frames_per_video= 16, frame_dim=(16, 16, 1), 
        sliding_window=True, grayscale=True, debug=False
    ):
        self.ext = ext
        self.n_frames_per_video = n_frames_per_video
        self.frame_dim = frame_dim
        self.sliding_window=sliding_window
        self.grayscale=grayscale


    def image_rgb_to_grayscale(self, image):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return np.expand_dims(img, axis=-1)
        
    def resize_image(self, image, dim):
        return cv2.resize(image, dim,interpolation=cv2.INTER_AREA)

    def count_frames(self, path):
        cap = cv2.VideoCapture(path)
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def get_video(self, video_path):
        return cv2.VideoCapture(video_path)

    def get_cropped_frames(self, video, grayscale=True, start_frame=None):
        if(start_frame is not None):
            video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        else:
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)

        frames = []
        for _k in range(self.n_frames_per_video):
            _, frame = video.read()
            img = self.resize_image(frame, self.frame_dim)

            if(grayscale == True):
                img = self.image_rgb_to_grayscale(img)

            frames.append(img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        return np.array(frames)

    def get_np_from_video(self, path):
        cap = cv2.VideoCapture(path)
        
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        buf = np.empty((frameCount, self.frame_dim[0], 
        self.frame_dim[1], self.frame_dim[2]), np.dtype('uint8'))

        i = 0
        success, frame = cap.read()
        
        while success:
            img = self.resize_image(frame, self.frame_dim)

            if(self.grayscale == True):
                img = self.image_rgb_to_grayscale(img)
                
            buf[i] = img
            success, frame = cap.read()
            
        return buf

    def get_window_from_video(self, video, start_frame=None):
        if(start_frame is None):
            return video[:self.n_frames_per_video]
        
        return video[start_frame:start_frame + self.n_frames_per_video]

    def get_clips(self, path, label):
        
        if(self.sliding_window == False):
            return Clip(path, label, 0, self.n_frames_per_video - 1)
        
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
            print(f"Returning clips with shape {np.shape(clips)}")
            
        return clips