import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import numpy as np
from scipy.interpolate import interp1d
from syncnet_python.detectors import S3FD

def load_face_detector(device):
    return S3FD(device)


def find_facetracks(face_detector, video, min_face_size=50):
    frame_bboxes = detect_faces_in_frames(face_detector, video.frames)
    for face_track in extract_face_tracks(frame_bboxes):
        if is_too_small(face_track, min_face_size):
            continue

        start_frame = face_track[0][0]
        end_frame = face_track[-1][0]

        yield video.trim(start_frame, end_frame).crop(interpolate_track(face_track))


def detect_faces_in_frames(face_detector, frames):
    face_detection_scales = [0.25]
    frame_bboxes = []

    for frame_id, frame in enumerate(frames):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_bboxes.append((frame_id, [tuple(face) for face in face_detector.detect_faces(frame, conf_th=0.9, scales=face_detection_scales)]))

    return frame_bboxes

def bb_intersection_over_union(bboxA, bboxB):
    xA1, yA1, xA2, yA2, _ = bboxA
    xB1, yB1, xB2, yB2, _ = bboxB

    interArea = max(0, min(xA2, xB2) - max(xA1, xB1)) * max(0, min(yA2, yB2) - max(yA1, yB1))
    boxAArea = (xA2 - xA1) * (yA2 - yA1)
    boxBArea = (xB2 - xB1) * (yB2 - yB1)
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

def extract_face_tracks(frame_bboxes):
    iouThres = 0.75    # Minimum IOU between consecutive face detections
    num_failed_det = 25
    min_track = 25

    tracks = []
    while True:
        track = []
        for frame_id, current_frame_bboxes in frame_bboxes:
            for bbox in current_frame_bboxes:
                if track == []:
                    track.append((frame_id, bbox))
                    current_frame_bboxes.remove(bbox)
                elif frame_id - track[-1][0] <= num_failed_det:
                    if bb_intersection_over_union(bbox, track[-1][1]) > iouThres:
                        track.append((frame_id, bbox))
                        current_frame_bboxes.remove(bbox)
                        continue
                else:
                    break

        if track == []:
            break

        if (track[-1][0] - track[0][0]) > min_track:
            yield track


def interpolate_track(track):
    frame_nums = np.array([ frame_num for frame_num, _ in track ])
    bboxes = np.array([np.array(bbox) for _, bbox in track])
    all_frame_nums = np.arange(frame_nums[0], frame_nums[-1] + 1)
    interpolated_bboxes = zip(*[
        list(interp1d(frame_nums, bboxes[:,i])(all_frame_nums))
        for i in range(4)
    ])

    return zip(all_frame_nums, list(interpolated_bboxes))

def is_too_small(track, min_face_size):
    bboxes = np.array([bbox for _, bbox in track])
    return max(np.mean(bboxes[:,2] - bboxes[:,0]), np.mean(bboxes[:,3] - bboxes[:,1])) <= min_face_size

