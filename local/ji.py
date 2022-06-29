import json
import torch
import sys
import numpy as np
import cv2
from pathlib import Path

# from ensemble_boxes import weighted_boxes_fusion

# from models.experimental import attempt_load
from utils.torch_utils import select_device, time_sync
from utils.general import check_img_size, non_max_suppression  # , scale_coords, xyxy2xywh
from utils.augmentations import letterbox

import argparse
import os

import torch.backends.cudnn as cudnn
from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages

from ByteTrack.yolox.tracker.byte_tracker import BYTETracker

device = torch.device("cuda:0")
# device = 'cpu'
model_path = '/project/train/models/train/exp2/weights/best.pt'  # 模型地址一定要和测试阶段选择的模型地址一致！！！


@torch.no_grad()
def init():
    weights = model_path
    device = 'cuda:0'  # cuda device, i.e. 0 or 0,1,2,3 or
    # device = 'cpu'
    device = select_device(device)
    half = True  # use FP16 half-precision inference
    data = '/project/ev_sdk/src/passenger.yaml'
    # w = str(weights[0] if isinstance(weights, list) else weights)
    # model = torch.jit.load(w) if 'torchscript' in w else attempt_load(weights) #, map_location=device)
    # if half:
    #     model.half()  # to FP16
    # model.eval()

    # Load model
    # device = select_device(device)

    model = DetectMultiBackend(weights, device=device, dnn=False, data=data, fp16=half)
    return model


def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Demo!")
    parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float, default=400, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    return parser


def process_video(handle=None, input_video=None, args=None, **kwargs):
    """Do inference to analysis input_video and get output
        输入： input_video='/path/to/input/video_seq0'
        输出： /path/to/output/tracker/tracker.txt
    Attributes:
    handle: algorithm handle returned by init()
    Returns: process result
    """
    if isinstance(args, str):
        output_tracker_file = args
    else:
        args = json.loads(args)
        output_tracker_file = args['output_tracker_file']
    # args = json.loads(args)
    # output_tracker_file = args['output_tracker_file']
    os.makedirs(os.path.dirname(output_tracker_file), exist_ok=True)
    # frames_dict = {}
    opt = make_parser().parse_args()
    tracker = BYTETracker(opt, frame_rate=opt.fps)
    results = []

    # for frame in pathlib.Path(input_video).glob('*.png'):
    #     frame_id = int(frame.with_suffix('').name)
    #     frames_dict[frame_id] = frame.as_posix()
    # frames = list(frames_dict.items())  # frames[¨] = (frame_id, frame_file)
    # frame_count = len(frames)

    # Process video here
    # Process each frame and generate a tracker file

    conf_thres = 0.25  # confidence threshold
    iou_thres = 0.45  # NMS IOU threshold
    max_det = 1000  # maximum detections per image
    classes = None  # filter by class: --class 0, or --class 0 2 3
    cls_names = ['head', 'white_XP_badge', 'blue_XP_badge', 'black_XP_badge', 'white_BH_badge', 'blue_BH_badge', 'male',
                 'female', '0-5', '6-10',
                 '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100']
    agnostic_nms = False
    imgsz = [1024, 1024]

    model = handle
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    # Dataloader
    dataset = LoadImages(input_video, img_size=imgsz, stride=stride, auto=pt)
    bs = 1  # batch_size

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, im, im0s, vid_cap, s in dataset:
        # frame_id = int(os.path.basename(path).split('.')[0])
        frame_id = int(s.split(' ')[1].split('/')[0])
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        pred = model(im, augment=False, visualize=False)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, multi_label=True,
                                   max_det=max_det)
        dt[2] += time_sync() - t3
        # pred[0][:,-1]
        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        det = pred[0]
        indices = (det[:, -1].int() == 0).nonzero().squeeze()
        outputs = torch.index_select(det[:, :5], 0, indices)
        if outputs is not None:
            online_targets = tracker.update(outputs.cpu(), im0s.shape, imgsz)  # xyxyc
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                if tlwh[2] * tlwh[3] > opt.min_box_area:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
                    # save results
                    results.append(
                        f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},1,1\n"
                    )

    with open(output_tracker_file, 'w') as tracker_file:
        # frame_id object_id x y width height confidence sex age
        # pred_tracker_data = '1,1,38,139,133,260,1,1,1'
        #  tracker_file.write(pred_tracker_data)

        tracker_file.writelines(results)

    print(results)
    return json.dumps({"model_data": {"objects": []}, "status": "success"})


if __name__ == '__main__':
    # Test API
    # img =cv2.imread('/home/data/831/helmet_10809.jpg')
    input_video = '/home/data/953'
    predictor = init()
    import time

    s = time.time()
    args = '/project/ev_sdk/src/tracker.txt'
    fake_result = process_video(predictor, input_video, args=args)
    e = time.time()
    print(fake_result)
    print((e - s))