# Audio-Visual Speech Recognition Pipeline
This repository contains a pipeline that can be used to extract talking faces from videos. The pipeline is based on [SyncNet](https://github.com/joonson/syncnet_python).

## Installation
```
bash install.sh
```

## Usage
```
Usage: run.py [OPTIONS] PATTERN OUTPUT_DIR

Options:
  --device TEXT                  CUDA device.  [default: cuda:0]
  --scene-threshold FLOAT        Threshold for histogram based shot detection.  [default: 0.004]
  --min-scene-duration INTEGER   Minimum scene duration in frames.  [default: 25]
  --min-face-size INTEGER        Minimum mean face size in pixels.  [default: 50]
  --detect-face-every-nth-frame INTEGER  Detect faces every nth frames.  [default: 1]
  --syncnet-threshold FLOAT      SyncNet threshold.  [default: 2.5]
  --min-speech-duration INTEGER  Minimum speech segment duration.  [default: 20]
  --max-pause-duration INTEGER   Maximum pause duration between speech
                                 segments.  [default: 10]
  --help                         Show this message and exit.  [default: False]
```
