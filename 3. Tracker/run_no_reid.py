import cv2
import os
import time
import pickle
import random
import argparse
import numpy as np
from trackers.tracker import Tracker


def write_results(filename, results):
    # Format: frame,id,x1,y1,w,h,score,class
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},{cls}\n'
    with open(filename, 'w') as f:
        for frame_id, track_ids, x1y1whs, scores, classes in results:
            for track_id, x1y1wh, score, cls in zip(track_ids, x1y1whs, scores, classes):
                x1, y1, w, h = x1y1wh
                line = save_format.format(frame=frame_id, id=track_id,
                                          x1=round(x1, 1), y1=round(y1, 1),
                                          w=round(w, 1), h=round(h, 1),
                                          s=round(score, 2), cls=cls)
                f.write(line)


def make_parser():
    parser = argparse.ArgumentParser("Tracker - No ReID, D-FINE detections")

    # Required: video and detections
    parser.add_argument("--video_path", type=str, required=True,
                        help="Path to the video file (used for FPS and to locate pickle files)")
    parser.add_argument("--pickle_dir", type=str,
                        default="/mnt/ext_hdd/sharath_track_bm/DATA/1_detection/",
                        help="Directory containing _nms06.pickle and _nms09.pickle files")
    parser.add_argument("--output_dir", type=str, default="../outputs/3. track/")
    parser.add_argument("--seed", type=float, default=10000)

    # Matching thresholds (tune per run)
    parser.add_argument("--det_thr", type=float, default=0.50,
                        help="Confidence threshold separating high/low detections")
    parser.add_argument("--init_thr", type=float, default=0.60,
                        help="Minimum confidence to spawn a new track")
    parser.add_argument("--match_thr", type=float, default=0.80,
                        help="Cost threshold for iterative assignment")

    # Tracker parameters
    parser.add_argument("--min_len", type=int, default=3)
    parser.add_argument("--min_box_area", type=float, default=200)
    parser.add_argument("--max_time_lost", type=float, default=30,
                        help="Fallback if FPS cannot be read from video")
    parser.add_argument("--penalty_p", type=float, default=0.20)
    parser.add_argument("--penalty_q", type=float, default=0.40)
    parser.add_argument("--reduce_step", type=float, default=0.05)
    parser.add_argument("--tai_thr", type=float, default=0.55)

    return parser


def get_video_stem(video_path):
    """Strip the final .mp4 extension to get the stem used in pickle filenames."""
    basename = os.path.basename(video_path)
    if basename.endswith('.mp4'):
        return basename[:-4]
    return basename


def run():
    # Derive video stem and locate pickle files
    video_stem = get_video_stem(args.video_path)
    pickle_nms09 = os.path.join(args.pickle_dir, video_stem + '_nms09.pickle')
    pickle_nms06 = os.path.join(args.pickle_dir, video_stem + '_nms06.pickle')

    if not os.path.exists(pickle_nms09):
        raise FileNotFoundError('Could not find: %s' % pickle_nms09)
    if not os.path.exists(pickle_nms06):
        raise FileNotFoundError('Could not find: %s' % pickle_nms06)

    # Read FPS from video and set max_time_lost = fps * 10
    cap = cv2.VideoCapture(args.video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps > 0:
        args.max_time_lost = int(fps * 10)
        print('Video FPS: %.1f  ->  max_time_lost: %d frames (10s)' % (fps, args.max_time_lost))
    else:
        print('Warning: could not read FPS from video, using max_time_lost=%d' % args.max_time_lost)

    # Load detections
    # _nms09: looser NMS -> more boxes -> main detections
    # _nms06: stricter NMS -> fewer boxes -> used for find_deleted_detections
    print('Loading detections for: %s' % video_stem)
    with open(pickle_nms09, 'rb') as f:
        detections = pickle.load(f)
    with open(pickle_nms06, 'rb') as f:
        detections_95 = pickle.load(f)

    # Make output folder
    os.makedirs(args.output_dir, exist_ok=True)

    # Track
    print('Tracking...')
    vid_name = list(detections.keys())[0]
    tracker = Tracker(args, vid_name)

    results = []
    total_time, total_count = 0, 0
    for frame_id in detections[vid_name].keys():
        start = time.time()
        if detections[vid_name][frame_id] is not None:
            track_results = tracker.update(detections[vid_name][frame_id],
                                           detections_95[vid_name][frame_id])
        else:
            track_results = tracker.update_without_detections()
        total_time += time.time() - start
        total_count += 1
        if total_count % 100 == 0:
            print('Processed %d / %d frames' % (total_count, len(detections[vid_name])))

        # Collect results
        x1y1whs, track_ids, scores, classes = [], [], [], []
        for t in track_results:
            if t.track_id > 0 and t.x1y1wh[2] * t.x1y1wh[3] > args.min_box_area:
                x1y1whs.append(t.x1y1wh)
                track_ids.append(t.track_id)
                scores.append(t.score)
                classes.append(t.cls)
        results.append([frame_id, track_ids, x1y1whs, scores, classes])

    # Write results
    result_filename = os.path.join(args.output_dir, '%s.txt' % vid_name)
    write_results(result_filename, results)
    print('Results saved to: %s' % result_filename)
    print('Speed: %.1f fps' % (total_count / total_time))


if __name__ == "__main__":
    args = make_parser().parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    run()
