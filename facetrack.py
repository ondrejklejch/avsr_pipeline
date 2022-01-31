import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import numpy as np
from scipy.interpolate import interp1d

import torch
from syncnet_python.detectors import S3FD
from syncnet_python.detectors.s3fd.box_utils import nms_

def load_face_detector(device):
    return S3FD(device)


def find_facetracks(face_detector, video, min_face_size=50, every_nth_frame=1):
    frame_bboxes = list(detect_faces_in_frames(face_detector, video.frames, every_nth_frame=every_nth_frame))
    for face_track in extract_face_tracks(frame_bboxes):
        if is_too_small(face_track, min_face_size):
            continue

        start_frame = face_track[0][0]
        end_frame = face_track[-1][0]

        yield video.cut(start_frame, end_frame).crop(interpolate_track(face_track))


def detect_faces_in_frames(face_detector, images, every_nth_frame=1, conf_th=0.9, scales=[0.25], batch_size=256):
    images = images[::every_nth_frame]

    with torch.no_grad():
        num_images = len(images)
        bboxes = [[] for _ in range(num_images)]
        w, h = images[0].shape[1], images[0].shape[0]

        for s in scales:
            for batch_offset in range(0, num_images, batch_size):
                batch_images = images[batch_offset:batch_offset + batch_size]
                preprocessed_images = [preprocess_image(image, s) for image in batch_images]
                x = torch.stack([torch.from_numpy(img).to(face_detector.device) for img in preprocessed_images], axis=0)
                y = face_detector.net(x)

                for image_id, detections in enumerate(y.data, batch_offset):
                    scale = torch.Tensor([w, h, w, h])

                    for i in range(detections.size(0)):
                        for j, score in enumerate(detections[i, :, 0]):
                            if score <= conf_th:
                                break

                            pt = (detections[i, j, 1:] * scale).cpu().numpy()
                            bbox = (pt[0], pt[1], pt[2], pt[3], score)
                            bboxes[image_id].append(bbox)

        for image_id, image_bboxes in enumerate(bboxes):
            if len(image_bboxes) == 0:
                continue

            image_bboxes = np.vstack(image_bboxes)
            keep = nms_(image_bboxes, 0.1)
            yield (image_id * every_nth_frame, [tuple(bbox) for bbox in image_bboxes[keep]])


def preprocess_image(image, scale):
    mean = np.array([123., 117., 104.])[:, np.newaxis, np.newaxis].astype('float32')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    return np.transpose(image, [2, 0, 1]).astype('float32') - mean


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

