import click
import glob
from video import load_video
from scene import segment_scenes

@click.command()
@click.option('--scene-threshold', default=0.004, help='Threshold for histogram based shot detection.')
@click.option('--min-scene-duration', default=25, help='Minimum scene duration in frames.')
@click.argument('pattern')
def main(scene_threshold, min_scene_duration, pattern):
    for path in glob.glob(pattern):
        video = load_video(path)
        for scene in segment_scenes(video, scene_threshold, min_scene_duration):
            pass


if __name__ == '__main__':
    main()
