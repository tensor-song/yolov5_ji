python /project/train/src_repo/yolov5/export.py --weights 
/project/train/models/train/exp/weights/best.pt --include onnx 

mo --input_model /project/train/models/train/exp/weights/best.onnx --data_type FP16 
--output_dir /project/train/models/train/exp/weights/
