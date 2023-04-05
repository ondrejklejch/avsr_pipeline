import os
import re
import click
import glob
from itertools import islice
from video import load_video
from scene import segment_scenes
from facetrack import load_face_detector, find_facetracks
from syncnet import load_syncnet, find_talking_segments

def load_segments(name):
    subtitles_path = "/disk/scratch1/s1569734/av2022/eval/%s.en.vtt" % name
    if not os.path.isfile(subtitles_path):
        print("Skipping %s because it doesn't have subtitles" % name)
        return []

    with open(subtitles_path, 'r') as f:
        is_inside = False
        for line in f:
            match = re.match(r"^(\d{2}):(\d{2}):(\d{2}\.\d+) --> (\d{2}):(\d{2}):(\d{2}\.\d+)$", line.strip())
            if match:
                if not is_inside:
                    start = int(match.group(1)) * 3600 + int(match.group(2)) * 60 + float(match.group(3))
                    end = int(match.group(4)) * 3600 + int(match.group(5)) * 60 + float(match.group(6))
                    text = ""
                    is_inside = True
                else:
                    end = int(match.group(4)) * 3600 + int(match.group(5)) * 60 + float(match.group(6))
            elif is_inside and line.strip() != "":
                text += " " + line.strip() 
            elif is_inside and line.strip() == "":
                if text[-1] in [".", "!", "?", '"', ")"]:
                    is_inside = False
                    segment_name = "%s-%03.1f-%03.1f" % (name, start, end)
                    yield segment_name, int(start * 25), int(end * 25 + 1), text.strip()


@click.command(context_settings=dict(show_default=True))
@click.option('--device', default='cuda:0', help='CUDA device.')
@click.option('--scene-threshold', default=0.004, help='Threshold for histogram based shot detection.')
@click.option('--min-scene-duration', default=25, help='Minimum scene duration in frames.')
@click.option('--min-face-size', default=50, help='Minimum mean face size in pixels.')
@click.option('--detect-face-every-nth-frame', default=3, help='Detect faces every nth frames.')
@click.option('--syncnet-threshold', default=2.0, help='SyncNet threshold.')
@click.option('--min-speech-duration', default=20, help='Minimum speech segment duration.')
@click.option('--max-pause-duration', default=20, help='Maximum pause duration between speech segments.')
@click.argument('pattern')
@click.argument('output_dir')
def main(device, scene_threshold, min_scene_duration, min_face_size, detect_face_every_nth_frame, syncnet_threshold, min_speech_duration, max_pause_duration, pattern, output_dir):
    face_detector = load_face_detector(device)
    syncnet = load_syncnet(device)

    with open(pattern, 'r') as f:
      paths = (l.strip() for l in f.readlines())

    for path in paths:
        name = path.split('/')[-1].rsplit('.', 1)[0]
        print("Processing %s" % name)

        video = load_video(path)
        for segment_name, start, end, text in load_segments(name):
            video.frames.prefetch(end)

            if (end - start) < 50:
                print("Skipping %s because it is shorter than 2 seconds" % segment_name)
                continue

            segment = video.cut(start, end)
            scenes = list(segment_scenes(segment, scene_threshold, min_scene_duration))
            if len(scenes) != 1:
                print("Skipping %s because it has %d scenes" % (segment_name, len(scenes)))
                continue

            facetracks = list(find_facetracks(face_detector, scenes[0], min_face_size, detect_face_every_nth_frame))
            if len(facetracks) != 1:
                print("Skipping %s because it contains %d facetracks" % (segment_name, len(facetracks)))
                continue
            
            talking_segments = list(find_talking_segments(syncnet, facetracks[0], syncnet_threshold, min_speech_duration, max_pause_duration))
            if len(talking_segments) != 1:
                print("Skipping %s because it contains %d talking segments" % (segment_name, len(talking_segments)))
                continue

            if len(talking_segments[0].frames) < 0.8 * (end - start):
                print("Skipping %s because it is shorter than 80%% of the original segment length" % segment_name)
                continue

            print("Successfully processed %s, writing to %s/%s.mp4" % (segment_name, output_dir, segment_name))
            talking_segments[0].write("%s/%s.mp4" % (output_dir, segment_name))


            new_start = talking_segments[0].frame_offset - segment.frame_offset
            new_end = new_start + len(talking_segments[0].frames)
            segment.cut(new_start, new_end).write("%s/%s.full.mp4" % (output_dir, segment_name))
            with open("%s/%s.txt" % (output_dir, segment_name), "w") as f:
              print(start // 25, end // 25, text, file=f)


if __name__ == '__main__':
    main()
