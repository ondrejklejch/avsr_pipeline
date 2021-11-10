import click
import glob
from video import load_video
from scene import segment_scenes
from facetrack import load_face_detector, find_facetracks

@click.command()
@click.option('--device', default='cuda:0', help='CUDA device.')
@click.option('--scene-threshold', default=0.004, help='Threshold for histogram based shot detection.')
@click.option('--min-scene-duration', default=25, help='Minimum scene duration in frames.')
@click.option('--min-face-size', default=50, help='Minimum mean face size in pixels.')
@click.argument('pattern')
def main(device, scene_threshold, min_scene_duration, min_face_size, pattern):
    face_detector = load_face_detector(device)

    for path in glob.glob(pattern):
        video = load_video(path)
        for scene in segment_scenes(video, scene_threshold, min_scene_duration):
            for facetrack in find_facetracks(face_detector, scene, min_face_size):
                pass


if __name__ == '__main__':
    main()
