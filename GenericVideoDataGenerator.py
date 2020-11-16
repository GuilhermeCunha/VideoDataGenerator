import tensorflow as tf
import skimage
import numpy as np
import math
from glob import glob
import os
import cv2
from copy import copy
from tqdm import tqdm
from .Clip import Clip
from .abstracts.ExtensionProcessor import ExtensionProcessor

from .VideoProcessor import VideoProcessor

import logging

logger = logging.getLogger(__name__)

class VideoDataGenerator(tf.keras.utils.Sequence):

    def __init__(
        self, dataset_path, processor, batch_size, 
        shuffle=True, class_names=None, debug=False, 
        pre_process=True,name="GENERATOR", cache_videos=False,
        max_cached_iters=0
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
        self.cache_videos = cache_videos
        
        self.class_names = class_names if class_names is not None else os.listdir(dataset_path)
        self.class_names.sort()
        self.n_classes = len(self.class_names)
        
        self.class_indexes = {}
        
        for index, class_name in enumerate(self.class_names):
            self.class_indexes[class_name] = index
        
        self.data = []
        self.train_data = []
        self.val_data = []
        self.test_data = []

        self.cached_videos = {}
        self.cache_counter = 0
        self.max_cached_iters = max_cached_iters
                
        self.__recognize_dataset__()
        self.indexes = np.arange(len(self.data))
        self.shuffle_data()


        logger.info(f"Dataset successfully recognized")
        logger.info(f"{np.shape(self.data)}")

    def __recognize_dataset__(self):
        data = []
        progress_bar = tqdm(self.class_names)
        for i, class_name in enumerate(progress_bar):
            label = self.class_indexes[class_name]
            pattern = os.path.join(self.dataset_path , class_name, f"*{self.processor.ext}")
            videos_paths = glob(pattern)
            
            if(self.debug == True and i == 0):
                logger.debug(f"Number of videos of class {class_name}: {len(videos_paths)}")
                
            for video_path in videos_paths:
                clips = self.processor.get_clips(video_path, label)

                data.extend(clips)
        
        self.data = np.array(data)
        
        logger.debug(f"[{self.name}] {len(data)} data were recognized from {self.dataset_path}")
        logger.debug(f"[{self.name}] Shape of data: {np.shape(data)} ")
        logger.debug(f"[{self.name}] Shape of data[0]: {np.shape(data[0])} ")

    def shuffle_data(self):
        logger.info(f"Shuffling data")
 
        np.random.shuffle(self.indexes)

    def on_epoch_end(self):
        if self.shuffle == True:
            self.shuffle_data()
            
    def __len__(self):
        return math.ceil(len(self.data) / self.batch_size)
        # return int(np.floor(len(self.data) / float(self.batch_size)))

    def preprocess_data(self, data):
        data = np.array(data).astype('float32')         
        
        data_mean = np.mean(data)
        
        if(data_mean != 0):
            data_max = np.max(data)

            data -= data_mean
            data /= data_max
        
        return data

    def get_y_true(self):
        y_true = []

        for x in range(self.__len__()):
            batch = self.__getitem__(x)[1]
            for y in range(self.batch_size):
                y_true.append(batch[y])

        return y_true
    
    def get_video(self, path):
        try:
            if(self.cache_videos == True):
                if(path in self.cached_videos):
                    return self.cached_videos[path]
                else:
                    video = self.processor.get_video(path)
                    self.cached_videos[path] = video

                    return video

            return self.processor.get_video(path)
        except Exception as e:
            logger.info('Error in get_video')
            logger.error(e)


    def clear_cache(self):
        self.cached_videos = {}
        
    def handle_cache(self):
        if(self.cache_videos == True):
            if(self.cache_counter > self.max_cached_iters):
                self.clear_cache()
                self.cache_counter = 0
            else:
                self.cache_counter = self.cache_counter + 1

    def __getitem__(self, index): # Generate one batch of data
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        batch_data = [self.data[i] for i in indexes]
        
        X = []
        y = []
        
        for clip in batch_data:
            try:
                video = self.get_video(clip.video_path)
                frames = self.processor.get_cropped_frames(video, start_frame=clip.start_frame)
                
                if(self.pre_process == True):
                    frames = self.preprocess_data(frames)
                    
                X.append(frames)
                y.append(clip.label)
                
            except Exception as e:
                logger.info(f"[{self.name}] __getitem__ ERROR {clip.video_path} | {clip.start_frame}")
                logger.error(e)
        
        categorical_y = tf.keras.utils.to_categorical(y, self.n_classes)

        self.handle_cache()
        return np.array(X), categorical_y
    

    def get_batch(self, index):
        return self.__getitem__(index)