# Audio-Visual Speech Recognition Pipeline
This repository contains a pipeline that can be used to extract talking faces from videos. The pipeline is based on [SyncNet](https://github.com/joonson/syncnet_python).

## Installation
```
bash install.sh
```

## Usage
```
Usage: run.py [OPTIONS] PATTERN

Options:
  --device TEXT                  CUDA device.
  --scene-threshold FLOAT        Threshold for histogram based shot detection.
  --min-scene-duration INTEGER   Minimum scene duration in frames.
  --min-face-size INTEGER        Minimum mean face size in pixels.
  --syncnet-threshold FLOAT      SyncNet threshold.
  --min-speech-duration INTEGER  Minimum speech segment duration.
  --max-pause-duration INTEGER   Maximum pause duration between speech
                                 segments.
  --help                         Show this message and exit.
```
