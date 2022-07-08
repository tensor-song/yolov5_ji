import json
import torch
import sys
import numpy as np
import cv2
from pathlib import Path

# from ensemble_boxes import weighted_boxes_fusion

# from models.experimental import attempt_load
from utils.torch_utils import select_device, time_sync
from utils.general import check_img_size, non_max_suppression, scale_coords  # , xyxy2xywh
from utils.augmentations import letterbox

import argparse
import os

import torch.backends.cudnn as cudnn
from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages

# from ByteTrack.yolox.tracker.byte_tracker import BYTETracker

device = torch.device("cuda:0")
# device = 'cpu'
model_path = '/project/train/models/exp/weights/last.pt'  # 模型地址一定要和测试阶段选择的模型地址一致！！！


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


# 输入的图像不做resize操作
def process_image_free(handle=None, input_image=None, args=None, **kwargs):
    """Do inference to analysis input_image and get output

    Attributes:

    handle: algorithm handle returned by init()
    input_image (numpy.ndarray): image to be process, format: (h, w, c), BGR

    Returns: process result

    """

    # Process image here
    # args = json.loads(args)
    # output_tracker_file = args['output_tracker_file']
    # os.makedirs(os.path.dirname(output_tracker_file), exist_ok=True)

    conf_thres = 0.25  # confidence threshold
    iou_thres = 0.45  # NMS IOU threshold
    max_det = 1000  # maximum detections per image
    classes = None  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms = False
    names = ['goggles', 'glasses', 'sunglasses', 'front_head', 'side_head', 'back_head']
    # Convert
    im = input_image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)

    # im=input_image

    im = torch.from_numpy(im).to(device)
    im = im.half() if handle.fp16 else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim

    # Inference
    pred = handle(im, augment=False, visualize=False)

    # NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, multi_label=True,
                               max_det=max_det)

    fake_result = {
        'objects': []
    }

    # Process predictions
    for i, det in enumerate(pred):  # per image
        if len(det):
            # Write results
            for *xyxy, conf, clss in reversed(det):
                fake_result['objects'].append({
                    'xmin': float(xyxy[0]),
                    'ymin': float(xyxy[1]),
                    'xmax': float(xyxy[2]),
                    'ymax': float(xyxy[3]),
                    'name': names[int(clss)],
                    'confidence': float(conf)
                })
    return json.dumps(fake_result, indent=4)


def process_image(handle=None, input_image=None, args=None, **kwargs):
    """Do inference to analysis input_image and get output

    Attributes:

    handle: algorithm handle returned by init()
    input_image (numpy.ndarray): image to be process, format: (h, w, c), BGR

    Returns: process result

    """

    # Process image here
    # args = json.loads(args)
    # output_tracker_file = args['output_tracker_file']
    # os.makedirs(os.path.dirname(output_tracker_file), exist_ok=True)

    conf_thres = 0.25  # confidence threshold
    iou_thres = 0.45  # NMS IOU threshold
    max_det = 1000  # maximum detections per image
    classes = None  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms = False
    names = ['goggles', 'glasses', 'sunglasses', 'front_head', 'side_head', 'back_head']

    # Padded resize
    img0 = input_image
    img_size = [1280, 1280]
    stride, pt = model.stride, model.pt
    im = letterbox(img0, img_size, stride=stride, auto=pt)[0]
    # Convert
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)

    # im=input_image

    im = torch.from_numpy(im).to(device)
    im = im.half() if handle.fp16 else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim

    # stride, names, pt = handle.stride, handle.names, handle.pt
    # imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Inference
    pred = handle(im, augment=False, visualize=False)

    # NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, multi_label=True,
                               max_det=max_det)

    fake_result = {
        'objects': []
    }

    # Process predictions
    for i, det in enumerate(pred):  # per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], img0.shape).round()

            # Write results
            for *xyxy, conf, clss in reversed(det):
                fake_result['objects'].append({
                    'xmin': float(xyxy[0]),
                    'ymin': float(xyxy[1]),
                    'xmax': float(xyxy[2]),
                    'ymax': float(xyxy[3]),
                    'name': names[int(clss)],
                    'confidence': float(conf)
                })
    result = {}
    result['algorithm_data'] = {
        "is_alert": True if len(fake_result['objects']) > 0 else False,
        "target_count": len(fake_result['objects']),
        "target_info": fake_result['objects']
    }
    result['model_data'] = fake_result
    return json.dumps(result, indent=4)


if __name__ == '__main__':
    # Test API
    # img =cv2.imread('/home/data/831/helmet_10809.jpg')
    input_video = '/home/data/679'
    predictor = init()
    # import time

    # s = time.time()
    # args = '/project/ev_sdk/src/tracker.txt'
    # fake_result = process_video(predictor, input_video, args=args)
    # e = time.time()
    model = predictor
    imgsz = [1280, 1280]
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    # Dataloader
    dataset = LoadImages(input_video, img_size=imgsz, stride=stride, auto=pt)
    # args = '/project/ev_sdk/src/tracker.txt'
    # fake_result = process_video(predictor, input_video, args=args)
    for path, im, im0s, vid_cap, s in dataset:
        fake_result = process_image(predictor, im)
    print(fake_result)
    # print((e - s))