import cv2
import numpy as np

def segment_scenes(segment, threshold, min_scene_length):
    scores = compute_scores(segment)

    prev_split = 0
    for i, score in enumerate(scores):
        i += 1
        if score > threshold and (i - prev_split) > min_scene_length:
            yield segment.trim(prev_split, i)
            prev_split = i

    if (i - prev_split) > min_scene_length:
        yield segment.trim(prev_split, i)


def histogram(img):
    hist = np.concatenate([
        cv2.calcHist([img[:,:,c]], [0], None, histSize=[16], ranges=[0, 256])
        for c in [0, 1, 2]
    ]).flatten()
    return hist / np.sum(hist)
