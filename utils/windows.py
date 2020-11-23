def get_sliding_windows(n_frames, size):
    start = 0
    stop = start + size

    windows = []
    
    while(stop <= n_frames):
        windows.append((start, stop))
        start += 1
        stop += 1

    return windows

def get_windows(n_frames, size, distance=0):
    start = 0
    stop = start + size

    windows = []
    
    while(stop <= n_frames):
        windows.append((start, stop))
        start = stop + distance
        stop = start + size

    return windows