import tensorflow as tf
import skimage
import numpy as np
import math
from glob import glob
import os
import cv2
from copy import copy
from tqdm import tqdm
from Clip import Clip
from abstracts.ExtensionProcessor import ExtensionProcessor

class VideoDataGenerator(tf.keras.utils.Sequence):

    def __init__(
        self, dataset_path, processor, batch_size, 
        shuffle=True, class_names=None, debug=False, 
        pre_process=True,name="GENERATOR"
        ):

        if not isinstance(processor, ExtensionProcessor):
            raise ValueError("'processor' must be an instance of ExtensionProcessor")
        
        self.processor = processor
        self.name = name
        self.batch_size = batch_size
        self.dataset_path = dataset_path
        self.shuffle = shuffle
        self.debug = debug
        self.pre_process = pre_process
        
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
    
    # def get_video_npy(self, path):
    #     return np.load(path)

    def __recognize_dataset__(self):
        data = []
        
        for class_name in self.class_names:
            label = self.class_indexes[class_name]
            pattern = os.path.join(self.dataset_path , class_name, f"*{self.processor.ext}")
            
            videos_paths = glob(pattern)
            for video_path in videos_paths:
                clips = self.processor.get_clips(video_path, label)

                data.extend(clips)
        
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
                video = self.processor.get_video(clip.video_path)
                frames = self.processor.get_cropped_frames(video, start_frame=clip.start_frame, grayscale=self.grayscale)

                if(self.pre_process == True):
                    frames = frames.astype('float32')
                    frames -= np.mean(frames)
                    frames /= np.max(frames)

                X.append(frames)
                y.append(clip.label)
            except:
                print(f"[{self.name}] ERROR {clip.video_path} | {clip.start_frame}")
        
        return np.array(X), tf.keras.utils.to_categorical(y, self.n_classes)
    