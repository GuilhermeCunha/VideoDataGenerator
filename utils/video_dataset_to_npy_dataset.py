import os
from tqdm import tqdm
from glob import glob
import numpy as np
import cv2

def create_folder(self, path):
    os.makedirs(path, exist_ok = True)

def get_np_from_video(self, path, frame_dim):
        cap = cv2.VideoCapture(path)
        
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        buf = np.empty((frameCount, frame_dim[0], frame_dim[1], frame_dim[2]), np.dtype('uint8'))

        i = 0
        success, frame = cap.read()
        
        while success:
            img = self.resize_image(frame, self.frame_dim)
            if(self.grayscale == True):
                img = self.image_rgb_to_grayscale(img)
                
            buf[i] = img
            success, frame = cap.read()
            
        return buf

def process_videos_to_npy(self, dataset_path, class_names, frame_dim=(16, 16, 1)):
        progress_bar = tqdm(self.class_names)
        
        for class_name in progress_bar:
            pattern = os.path.join(self.dataset_path , class_name, f"*{self.video_ext}")
            
            videos_paths = glob(pattern)

            for video_path in videos_paths:
                npy = self.get_np_from_video(video_path, frame_dim=frame_dim)
                destination = os.path.relpath(video_path, self.dataset_path)
                class_name =  os.path.dirname(destination)
                self.create_folder(os.path.join(self.processed_dataset_root_folder, class_name))
                dest = f"{os.path.join(self.processed_dataset_root_folder, os.path.splitext(destination)[0])}.npy"
                
                with open(dest, 'wb') as f:
                    np.save(f, npy)