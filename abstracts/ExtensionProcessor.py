from abc import ABC, abstractmethod

class ExtensionProcessor(ABC):
    def __init__(self, ext):
        self.ext = ext

    @abstractmethod
    def get_video(self, path):
        pass
        
    @abstractmethod
    def get_window_from_video(self, video, start_frame=None):
        pass
    
    @abstractmethod
    def get_cropped_frames(self, video, grayscale=True, start_frame=None):
        pass
    
    @abstractmethod
    def count_frames(self, path):
        pass