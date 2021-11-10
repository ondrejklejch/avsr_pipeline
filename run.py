import click
import glob
from video import load_video
from scene import segment_scenes
from facetrack import load_face_detector, find_facetracks
from syncnet import load_syncnet, find_talking_segments

@click.command()
@click.option('--device', default='cuda:0', help='CUDA device.')
@click.option('--scene-threshold', default=0.004, help='Threshold for histogram based shot detection.')
@click.option('--min-scene-duration', default=25, help='Minimum scene duration in frames.')
@click.option('--min-face-size', default=50, help='Minimum mean face size in pixels.')
@click.option('--syncnet-threshold', default=2.5, help='SyncNet threshold.')
@click.option('--min-speech-duration', default=20, help='Minimum speech segment duration.')
@click.option('--max-pause-duration', default=10, help='Maximum pause duration between speech segments.')
@click.argument('pattern')
def main(device, scene_threshold, min_scene_duration, min_face_size, syncnet_threshold, min_speech_duration, max_pause_duration, pattern):
    face_detector = load_face_detector(device)
    syncnet = load_syncnet(device)

    for path in glob.glob(pattern):
        name = path.split('/')[-1].rsplit('.', 1)[0]
        print("Processing %s" % name)

        video = load_video(path)
        scenes = segment_scenes(video, scene_threshold, min_scene_duration)
        for i, scene in enumerate(scenes):
            facetracks = find_facetracks(face_detector, scene, min_face_size)
            for j, facetrack in enumerate(facetracks):
                segments = find_talking_segments(syncnet, facetrack, syncnet_threshold, min_speech_duration, max_pause_duration)
                for k, segment in enumerate(segments):
                    segment.write('output/%s-scene_%d-facetrack_%d-segment_%d.mp4' % (name, i, j, k))


if __name__ == '__main__':
    main()
