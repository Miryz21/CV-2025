import argparse
from tracker import VideoTracker


def main():
    parser = argparse.ArgumentParser(description='Main script to run tracker.')
    parser.add_argument('--path', type=str, help='Path to video file')
    parser.add_argument('--out-dir', type=str, default=None, help='Path to out direcory')
    args = parser.parse_args()

    video_path = args.path
    if video_path is None:
        raise RuntimeError("Video path should be provided")

    tracker = VideoTracker(video_path, output_dir=args.out_dir)
    tracker.run_tracking()


if __name__ == "__main__":
    main()
