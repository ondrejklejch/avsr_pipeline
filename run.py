import click
import glob
from video import load_video

@click.command()
@click.argument('pattern')
def main(pattern):
    for path in glob.glob(pattern):
        video = load_video(path)


if __name__ == '__main__':
    main()
