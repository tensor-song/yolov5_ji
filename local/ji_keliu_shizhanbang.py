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
from utils.metrics import box_iou

import argparse
import os

import torch.backends.cudnn as cudnn
from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages

from ByteTrack.yolox.tracker.byte_tracker import BYTETracker

device = torch.device("cuda:0")
# device = 'cpu'
model_path = '/project/train/models/train/exp/weights/last.pt'  # 模型地址一定要和测试阶段选择的模型地址一致！！！


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
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    return parser


def in_out(cross_line, pt):
    # temp = (y1 – y2) * x + (x2 – x1) * y + x1 * y2 – x2 * y1
    temp = (cross_line[0, 1] - cross_line[1, 1]) * pt[0] + (cross_line[1, 0] - cross_line[0, 0]) * pt[1] + cross_line[
        0, 0] * cross_line[1, 1] - cross_line[1, 0] * cross_line[0, 1]
    return temp


def process_video(handle=None, input_video=None, args=None, **kwargs):
    """Do inference to analysis input_video and get output
        输入： input_video='/path/to/input/video_seq0'
        输出： /path/to/output/tracker/tracker.txt
    Attributes:
    handle: algorithm handle returned by init()
    Args:{
    "roi": "POLYGON((0.63537 0.22692,0.83624 0.37692,0.01747 0.82692,0.01310 0.48077))",//统计区域的ROI
    "cross_line":   "LINESTRING(0.00218 0.65000,0.76419 0.27692)",
    //穿过该线则视为进入或出去
    "end":  "POINT(0.38646 0.52692)", //相对位置，表示出的方向
    "name": "1.mp4",
    }
    Returns: process result
    """

    # if isinstance(args, str):
    # output_tracker_file = args
    # else:
    # args = json.loads(args)
    # output_tracker_file = args['output_tracker_file']
    # output_tracker_file = args

    args = json.loads(args)
    temp = args['roi'].split('((')[1].split('))')[0].split(',')
    roi = np.array([i.split(' ') for i in temp], dtype=float)
    # roi = roi.astype(float)
    cross_line = np.array([i.split(' ') for i in args['cross_line'].split('(')[1].split(')')[0].split(',')],
                          dtype=float)
    end = np.array(args['end'].split('(')[1].split(')')[0].split(' '), dtype=float)

    # if args['is_use_card'] is not None:
    #     is_use_card = args['is_use_card']
    # else:
    #     is_use_card = True #是否开启工牌去重

    # if args['alert_card'] is not None:
    #     alert_card = args['alert_card']
    # else:
    #     alert_card = ['white_XP_badge', 'blue_XP_badge', 'black_XP_badge', 'white_BH_badge', 'blue_BH_badge']     #工牌识别的开关，true时打开识别工牌，false时关闭识别工牌，工牌列表见附1

    # if args['is_count'] is not None:
    #     is_count = args['is_count']
    # else:
    #     is_count = True #当为true时，统计”enter”与”exit“数量，当为false时，不进行统计”enter”与”exit“数量

    is_use_card = True  # 是否开启工牌去重
    alert_card = ['white_XP_badge', 'blue_XP_badge', 'black_XP_badge', 'white_BH_badge',
                  'blue_BH_badge']  # 工牌识别的开关，true时打开识别工牌，false时关闭识别工牌，工牌列表见附1
    is_count = True

    # output_tracker_file = args['output_tracker_file']

    # os.makedirs(os.path.dirname(output_tracker_file), exist_ok=True)
    # frames_dict = {}
    opt = make_parser().parse_args()
    tracker = BYTETracker(opt, frame_rate=opt.fps)
    results = []

    # Process video here
    # Process each frame and generate a tracker file
    conf_thres = 0.25  # confidence threshold
    iou_thres = 0.45  # NMS IOU threshold
    max_det = 1000  # maximum detections per image
    classes = None  # filter by class: --class 0, or --class 0 2 3
    # cls_names = ['head', 'white_XP_badge', 'blue_XP_badge', 'black_XP_badge', 'white_BH_badge', 'blue_BH_badge', 'male',
    #  'female', '0-5', '6-10',
    #  '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100']
    cls_names = ['head', 'white_XP_badge', 'blue_XP_badge', 'black_XP_badge', 'white_BH_badge', 'blue_BH_badge', 'male',
                 'female']
    agnostic_nms = False
    imgsz = [1280, 1280]

    model = handle
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    # Dataloader
    dataset = LoadImages(input_video, img_size=imgsz, stride=stride, auto=pt)
    bs = 1  # batch_size

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    # dt, seen = [0.0, 0.0, 0.0], 0

    fake_result = {}
    target_info = []
    objects = []
    # flag = {'trck_id':{'is_enter':False, 'is_exits':False, 'card':None}}
    flag = {}
    for path, im, im0s, vid_cap, s in dataset:
        # frame_id = int(os.path.basename(path).split('.')[0])
        frame_id = int(s.split(' ')[1].split('/')[0])
        w, h, _ = im0s.shape
        cross_line[:, 0] = cross_line[:, 0] * w
        cross_line[:, 1] = cross_line[:, 1] * h
        roi[:, 0] = roi[:, 0] * w
        roi[:, 1] = roi[:, 1] * h
        # roi = np.array(roi, dtype=int)
        roi = roi.astype(int)
        end = (end[0] * w, end[1] * h)

        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        # Inference
        pred = model(im, augment=False, visualize=False)

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, multi_label=True,
                                   max_det=max_det)

        # Process predictions
        det = pred[0]
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0s.shape).round()

        indices = (det[:, -1].int() == 0).nonzero().squeeze()
        outputs = torch.index_select(det[:, :5], 0, indices)

        indices_others = (det[:, -1].int() != 0).nonzero().squeeze()
        outputs_others = torch.index_select(det, 0, indices_others)
        indices_sex = (det[:, -1].int() > 5).nonzero().squeeze()
        outputs_sex = torch.index_select(det, 0, indices_sex)
        indices_card = (outputs_others[:, -1].int() < 6).nonzero().squeeze()
        outputs_card = torch.index_select(det, 0, indices_card)

        # if outputs is not None:

        # online_targets = tracker.update(outputs.cpu(), im0s.shape, imgsz)  # xyxyc
        online_targets = tracker.update(outputs.cpu(), im0s.shape, im0s.shape)  # xyxyc
        online_tlwhs = []
        online_ids = []
        online_scores = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            if tlwh[2] * tlwh[3] > opt.min_box_area:
                # save results
                # results.append(
                # f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},1,1\n"
                # )
                pt = (int(tlwh[0]) + int(tlwh[2]) / 2, int(tlwh[1]) + int(tlwh[3]) / 2)
                if cv2.pointPolygonTest(roi, pt, measureDist=False) >= 0:
                    if outputs_sex.shape[0] > 0:
                        ious = box_iou(torch.tensor(t.tlbr), outputs_sex[:, :4])
                        sex = outputs_sex[ious.argmax(dim=1, keepdim=True)]
                    else:
                        sex = 'female'
                    base_msg = {
                        "x": int(tlwh[0]),
                        "y": int(tlwh[1]),
                        "height": int(tlwh[3]),
                        "width": int(tlwh[2]),
                        "confidence": float(t.score),
                        "person_id": str(t.track_id),
                        "sex": sex,
                        "age": np.random.randint(18, 50),
                    }
                    objects.append(base_msg)

                    if str(tid) not in flag.keys():
                        flag[str(tid)] = {'is_enter': False, 'is_exits': False, 'tag': None}
                    # Tmp = (y1 – y2) * x + (x2 – x1) * y + x1 * y2 – x2 * y1
                    # Tmp < 0 在左侧
                    # Tmp = 0 在线上
                    # Tmp > 0 在右侧
                    is_end = in_out(cross_line, pt)
                    ended = in_out(cross_line, end)
                    if not (is_end > 0) ^ (ended > 0):
                        if flag[str(tid)]['tag'] is None:
                            flag[str(tid)]['tag'] = False
                        elif flag[str(tid)]['tag'] is True:
                            if not (flag[str(tid)]['is_exits'] or flag[str(tid)]['is_enter']):
                                flag[str(tid)]['is_exits'] = True
                                flag[str(tid)]['sex'] = sex

                    else:
                        if flag[str(tid)]['tag'] is None:
                            flag[str(tid)]['tag'] = True
                        elif flag[str(tid)]['tag'] is False:
                            if not (flag[str(tid)]['is_exits'] or flag[str(tid)]['is_enter']):
                                flag[str(tid)]['is_enter'] = True
                                flag[str(tid)]['sex'] = sex

                    if is_use_card:
                        for outcard in outputs_card:
                            if cls_names[int(outcard[-1])] in alert_card:
                                center = int((outcard[2] + outcard[0]) / 2)
                                # card 绑定人
                                if int(tlwh[0]) < center and center < (int(tlwh[0]) + int(tlwh[2])):
                                    base_msg["card"] = cls_names[int(outcard[-1])]
                                    flag[str(tid)]['is_enter'] = False
                                    flag[str(tid)]['is_exits'] = False

                    base_msg["enter"] = flag[str(tid)]['is_enter']
                    base_msg["exit"] = flag[str(tid)]['is_exits']
                    if flag[str(tid)]['is_enter'] or flag[str(tid)]['is_exits']:
                        target_info.append(base_msg)

        # if is_use_card：
        #     for outcard in outputs_card:
        #         if cls_names[int(outcard[-1])] in alert_card:
        #             objects.append({
        #                     "x": int(outcard[0]),
        #                     "y": int(outcard[1]),
        #                     "height": int(outcard[3]-outcard[1]),
        #                     "width": int(outcard[2]-outcard[0]),
        #                     "confidence":outcard[4],
        #                     "card": cls_names[int(cls_names[-1])]   #当检测到工牌时，模型输出其识别的细分种类,细分种类见附1
        #                 })

        for outcard in outputs_card:
            pt = ((int(outcard[0]) + int(outcard[2])) / 2, (int(outcard[1]) + int(outcard[3])) / 2)
            if cv2.pointPolygonTest(roi, pt, measureDist=False) >= 0:
                objects.append({
                    "x": int(outcard[0]),
                    "y": int(outcard[1]),
                    "height": int(outcard[3] - outcard[1]),
                    "width": int(outcard[2] - outcard[0]),
                    "confidence": float(outcard[4]),
                    "card": cls_names[int(outcard[-1])]  # 当检测到工牌时，模型输出其识别的细分种类,细分种类见附1
                })

    # with open(output_tracker_file, 'w') as tracker_file:
    # frame_id object_id x y width height confidence sex age
    # pred_tracker_data = '1,1,38,139,133,260,1,1,1'
    #  tracker_file.write(pred_tracker_data)
    # tracker_file.writelines(results)

    enters = len([flag[id]['is_enter'] for id in flag.keys() if flag[id]['is_enter'] is True])
    exits = len([flag[id]['is_exits'] for id in flag.keys() if flag[id]['is_exits'] is True])

    enter_male = len(
        [flag[id]['sex'] for id in flag.keys() if flag[id]['is_enter'] is True and flag[id]['sex'] is 'male'])
    exit_male = len(
        [flag[id]['sex'] for id in flag.keys() if flag[id]['is_exits'] is True and flag[id]['sex'] is 'male'])
    enter_female = len(
        [flag[id]['sex'] for id in flag.keys() if flag[id]['is_enter'] is True and flag[id]['sex'] is 'female'])
    exit_female = len(
        [flag[id]['sex'] for id in flag.keys() if flag[id]['is_exits'] is True and flag[id]['sex'] is 'female'])
    fake_result["algorithm_data"] = {
        "is_alert": True if len(target_info) > 0 else False,
        "target_count": len(target_info),
        "is_use_card": is_use_card,
        "alert_card": alert_card,
        "is_count": is_count,
        "enter": enters,
        "exit": exits,
        "enter_male": enter_male,
        "exit_male": exit_male,
        "enter_female": enter_female,
        "exit_female": exit_female,
        "target_info": target_info
    }

    fake_result["model_data"] = {
        "objects": objects
    }
    return json.dumps(fake_result)
    # "target_info": [
    #     {
    #         "x": 716,
    #         "y": 716,
    #         "height": 646,
    #         "width": 233,
    #         "confidence": 0.999660,
    #         "person_id": "1",
    #         "sex": "female",
    #         "age": 20,
    #         "enter": true,
    #         "exit": false,

    #         "card": string   #算法会将工牌与人员进行捆绑，当发生报警时，输出其细分标签
    #     },

    # "objects": [
    #     {
    #         "x": 716,
    #         "y": 716,
    #         "height": 646,
    #         "width": 233,
    #         "confidence": 0.999660,
    #         "person_id": "1",
    #         "sex": "female",
    #         "age": 20,

    #         "card": string   #当检测到工牌时，模型输出其识别的细分种类,细分种类见附1
    #     },
    # print(results)
    # return json.dumps({"model_data": {"objects": []}, "status": "success"})


if __name__ == '__main__':
    # Test API
    # img =cv2.imread('/home/data/831/helmet_10809.jpg')
    input_video = '/home/data/953'
    input_video = '/project/train/src_repo/yolov5/1.mp4'
    predictor = init()
    import time

    s = time.time()
    args = '/project/ev_sdk/src/tracker.txt'
    args = {
        "roi": "POLYGON((0.63537 0.22692,0.83624 0.37692,0.01747 0.82692,0.01310 0.48077))",  # 统计区域的ROI
        "cross_line": "LINESTRING(0.00218 0.65000,0.76419 0.27692)",
        # 穿过该线则视为进入或出去
        "end": "POINT(0.38646 0.52692)",  # 相对位置，表示出的方向
        "name": "1.mp4",
    }
    fake_result = process_video(predictor, input_video, args=args)
    e = time.time()
    print(fake_result)
    print((e - s))