import os
import cv2
import glob
import numpy as np
from scipy.io import wavfile
from scipy.signal import medfilt
import subprocess


class Video:

    def __init__(self, frames, audio, audio_sample_rate, frame_offset = 0, parent=None):
        self.frames = frames
        self.audio = audio
        self.audio_sample_rate = audio_sample_rate
        self.frame_offset = frame_offset
        self.parent = parent

    def cut(self, start_frame, end_frame, offset=0):
        if start_frame >= offset:
            return Video(
                self.frames[start_frame:end_frame],
                self.audio[(start_frame - offset) * 640:(end_frame - offset) * 640],
                self.audio_sample_rate,
                self.frame_offset + start_frame,
                self
            )
        else:
            return Video(
                self.frames[(start_frame + offset):(end_frame + offset)],
                self.audio[start_frame * 640:end_frame * 640],
                self.audio_sample_rate,
                self.frame_offset + start_frame + offset,
                self
            )

    def trim(self):
        len_frames = len(self.frames)
        len_audio = len(self.audio) // 640

        if len_frames >= len_audio:
            return Video(
                self.frames[:len_audio],
                self.audio,
                self.audio_sample_rate,
                self.frame_offset,
                self
            )
        else:
            return Video(
                self.frames,
                self.audio[:len_frames * 640],
                self.audio_sample_rate,
                self.frame_offset,
                self
            )

    def crop(self, track, crop_padding_factor=0.4):
        xs = []
        ys = []
        sizes = []

        for _, (y1, x1, y2, x2) in track:
            xs.append((x1 + x2) / 2) # crop center x
            ys.append((y1 + y2) / 2) # crop center y
            sizes.append(max((x2 - x1), (y2 - y1)) / 2) # crop size

        # Smooth detections
        xs = medfilt(xs, kernel_size=13)
        ys = medfilt(ys, kernel_size=13)
        sizes = medfilt(sizes, kernel_size=13)

        cropped_frames = []
        for x, y, size, frame in zip(xs, ys, sizes, self.frames):
            padding = int(size * crop_padding_factor)
            padded_frame = np.pad(
                frame,
                ((padding, padding), (padding, padding), (0,0)),
                'constant',
                constant_values=(110,110)
            )

            face = padded_frame[
                max(0, int(x + padding - size)):int(x + size + 3 * padding), # 2 * size + 2 * padding
                max(0, int(y - size)):int(y + size + 2 * padding)            # 2 * size + 2 * padding
            ]

            cropped_frames.append(cv2.resize(face, (224,224)))

        return Video(
            cropped_frames,
            self.audio,
            self.audio_sample_rate,
            self.frame_offset,
            self.parent
        )

    def plot(self, title, rows=5, cols=4):
        num_frames = len(self.frames)
        shift = num_frames // (rows * cols - 1)

        fig, axs = plt.subplots(rows, cols)
        fig.set_size_inches(10, 1.7 * rows + 0.5)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.suptitle(title, fontsize=20)

        if shift == 0:
            frame_nums = range(num_frames)
        else:
            frame_nums = [i * shift for i in range(cols * rows - 1)] + [num_frames - 1]

        for i, frame_num in enumerate(frame_nums):
            if rows > 1:
                axis = axs[i // cols, i % cols]
            else:
                axis = axs[i]

            axis.set_xlabel('Frame %d, time: %.2fs' % (frame_num, frame_num * 0.04))
            axis.imshow(cv2.cvtColor(self.frames[frame_num], cv2.COLOR_BGR2RGB))
            axis.axis('off')

        plt.show()

    def write(self, path, tmpdir='/tmp'):
        tmp_wav = tmpdir + "/audio.wav"
        tmp_avi = tmpdir + "/video_without_audio.avi"
        wavfile.write(tmp_wav, self.audio_sample_rate, self.audio)

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        shape = (self.frames[0].shape[1], self.frames[0].shape[0])
        vOut = cv2.VideoWriter(tmp_avi, fourcc, 25, shape)
        for frame in self.frames:
            vOut.write(frame)
        vOut.release()

        command = ("ffmpeg -v 8 -y -i %s -i %s -strict -2 %s" % (tmp_wav, tmp_avi, path))
        output = subprocess.call(command, shell=True, stdout=None)

        os.remove(tmp_wav)
        os.remove(tmp_avi)


def load_video(path):
    video = load_frames(path)
    sample_rate, audio = load_audio(path)
    return Video(video, audio, sample_rate)


def load_frames(path):
    return Frames(path)


def load_audio(path):
    command = ("ffmpeg -v 8 -y -i %s -vn -acodec pcm_s16le -ar 16000 -ac 1 %s" % (path, path + ".wav"))
    output = subprocess.call(command, shell=True, stdout=None)
    sample_rate, audio = wavfile.read(path + ".wav")
    os.remove(path + ".wav")

    return sample_rate, audio


class Frames:

    def __init__(self, path):
        self.path = path
        self.frames = []
        self.offset = 0

    def __iter__(self):
        cap = cv2.VideoCapture(self.path)
        frame_num = 0
        segment_num = 0

        frames = []
        while True:
            ret, frame = cap.read()
            if ret == 0:
                break

            self.frames.append(frame)
            yield frame

    def __getitem__(self, key):
        if not isinstance(key, slice):
            raise ValueError('Class Frames supports only slicing')

        start = key.indices(len(self))[0]
        end = key.indices(len(self))[1]

        if start < self.offset or (end - start) > len(self.frames):
            raise ValueError('Trying to slice invalid frame')

        sliced_frames = self.frames[(start - self.offset):(end - self.offset)]
        self.frames = self.frames[(end - self.offset):]
        self.offset = end

        return sliced_frames

    def __len__(self):
        return self.offset + len(self.frames)
