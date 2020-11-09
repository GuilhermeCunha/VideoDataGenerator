import os
from tqdm import tqdm
from glob import glob
import numpy as np
import cv2

def create_folder(path):
    os.makedirs(path, exist_ok = True)

def image_rgb_to_grayscale(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.expand_dims(img, axis=-1)

def resize_image(image, dim):
    return cv2.resize(image, (dim[0], dim[1]),interpolation=cv2.INTER_AREA)

def get_np_from_video(path, frame_dim):
    cap = cv2.VideoCapture(path)
    
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    buf = np.empty((frameCount, frame_dim[0], frame_dim[1], frame_dim[2]), np.dtype('uint8'))

    i = 0
    success, frame = cap.read()
    
    while success:
        img = resize_image(frame, frame_dim)
        if(frame_dim[2] == 1):
            img = image_rgb_to_grayscale(img)
            
        buf[i] = img
        success, frame = cap.read()
        
    return buf

def process_videos_to_npy(dataset_path, output_path, class_names, frame_dim=(16, 16, 1), video_ext='.avi'):
    progress_bar = tqdm(class_names)
    
    for class_name in progress_bar:
        pattern = os.path.join(dataset_path , class_name, f"*{video_ext}")
        
        videos_paths = glob(pattern)

        for video_path in videos_paths:
            npy = get_np_from_video(video_path, frame_dim=frame_dim)
            destination = os.path.relpath(video_path, dataset_path)
            class_name =  os.path.dirname(destination)
            create_folder(os.path.join(output_path, class_name))
            dest = f"{os.path.join(output_path, os.path.splitext(destination)[0])}.npy"
            
            with open(dest, 'wb') as f:
                np.save(f, npy)