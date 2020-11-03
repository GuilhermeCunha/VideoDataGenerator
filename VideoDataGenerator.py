import tensorflow as tf
import skimage
import numpy as np
import math
from glob import glob
import os
import cv2
from copy import copy

class Clip:
    def __init__(self, video_path, label, start_frame, stop_frame):
        self.video_path = video_path
        self.start_frame = start_frame
        self.stop_frame = stop_frame
        self.label = label
        
class VideoDataGenerator(tf.keras.utils.Sequence):

    def __init__(self, dataset_path, batch_size, shuffle=True, class_names=None, 
                 n_frames_per_video=16, frame_dim=(16, 16), video_ext=".mp4", 
                 sliding_window=True, debug=False, cache_videos=False, 
                 grayscale=True, pre_process=True, name="GENERATOR"
                ):
        self.name = name
        self.batch_size = batch_size
        self.dataset_path = dataset_path
        self.sliding_window = sliding_window
        self.n_frames_per_video = n_frames_per_video
        self.frame_dim = frame_dim
        self.video_ext = video_ext
        self.shuffle = shuffle
        self.debug = debug
        self.grayscale = grayscale
        self.pre_process = pre_process
        self.cached_videos = {}
        self.cache_videos = cache_videos
        
        self.class_names = class_names if class_names is not None else os.listdir(dataset_path)
        self.n_classes = len(self.class_names)
        
        self.class_indexes = {}
        
        for index, class_name in enumerate(self.class_names):
            self.class_indexes[class_name] = index
        
        self.data = []
        self.train_data = []
        self.val_data = []
        self.test_data = []
        self.__recognize_dataset__()
        self.on_epoch_end()
        
    def resize_image(self, image, dim):
        return cv2.resize(image, dim,interpolation=cv2.INTER_AREA)
    
    def image_rgb_to_grayscale(self, image):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return np.expand_dims(img, axis=-1)
        
    
    def count_frames(self, path):
        cap = cv2.VideoCapture(path)
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    def cache_video(self, video, path):
        self.cached_videos[path] = video
    
    
    def read_video(self, path):
        if(self.cache_videos == True):
            if(path in self.cached_videos):
                return self.cached_videos[path]

            video = cv2.VideoCapture(path)
            self.cache_video(video, path)
            return video
            
        return cv2.VideoCapture(path)
    
    def get_cropped_frames(self, video, grayscale=True, start_frame=None):
        if(start_frame is not None):
            video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        else:
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)

        fps = video.get(5)

        frames = []
        for k in range(self.n_frames_per_video):
            ret, frame = video.read()
            img = self.resize_image(frame, self.frame_dim)

            if(grayscale == True):
                img = self.image_rgb_to_grayscale(img)

            frames.append(img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        return np.array(frames)
        
    
    
    def get_clips(self, path, label):
        
        if(self.sliding_window == False):
            return Clip(path, 0, self.n_frames_per_video - 1)
        
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
            
        return clips
    
    def __recognize_dataset__(self):
        data = []
        
        for class_name in self.class_names:
            label = self.class_indexes[class_name]
            
            pattern = os.path.join(self.dataset_path , class_name, f"*{self.video_ext}")
            
            videos_paths = glob(pattern)
            for video_path in videos_paths:
                clips = self.get_clips(video_path, label)
                for clip in clips:
                    data.append(clip)
        
        self.data = np.array(data)
        
        if(self.debug == True):
            print(f"{len(data)} data were recognized from {self.dataset_path}")
        
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.data))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def __len__(self):
        return math.ceil(len(self.data) / self.batch_size)

    def __getitem__(self, index): # Generate one batch of data
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        batch_data = [self.data[i] for i in indexes]
        
        X = []
        y = []
        
        for clip in batch_data:
            try:
                video = self.read_video(clip.video_path)
                frames = self.get_cropped_frames(video, start_frame=clip.start_frame, grayscale=self.grayscale)

                if(self.pre_process == True):
                    frames = frames.astype('float32')
                    frames -= np.mean(frames)
                    frames /= np.max(frames)

                X.append(frames)
                y.append(clip.label)
            except:
                print(f"[{self.name}] ERROR {clip.video_path} | {clip.start_frame}")
        
        return np.array(X), tf.keras.utils.to_categorical(y, self.n_classes)