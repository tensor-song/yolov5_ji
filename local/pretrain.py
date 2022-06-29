import torch

ckpt = torch.load('/work/slc/detection/yolov5/runs/train/person/weights/last.pt', map_location='cpu')
torch.save(ckpt['model'].float().state_dict(), '/work/slc/detection/yolov5/runs/train/person/weights/last2.pt')