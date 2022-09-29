# Yolov7 inference pipeline
### Pros
Ultra High speed kernel in GPU using:
 - NvJpeg
 - cupy
 - CUDA Module preprocessing ( Resize + Padding + Transpose + Normalise )
 - TensorRT

```bash
sudo apt install -y nvidia-driver-515
docker build -t vanjie_trt -f dockerfile_trt .
docker run --rm \
           --runtime=nvidia \
           --name=vanjie_trt \
           --privileged \
           -it \
           -v $PWD:/py/ \
           -w /py \
           vanjie_trt bash

# git clone https://github.com/WongKinYiu/yolov7
# git clone https://github.com/Monday-Leo/YOLOv7_Tensorrt.git
```


```bash
python /py/yolov7/export.py --weights /py/best.pt --batch-size 8 --grid --end2end --simplify --topk-all 100 --iou-thres 0.45 --conf-thres 0.25 --img-size 640 640
mv /py/best.onnx /py/ped_yolov7_8.onnx
trtexec --onnx=/py/ped_yolov7_64.onnx --saveEngine=/py/ped_yolov7_64.trt  --fp16 --workspace=8192 --explicitBatch --dumpOutput --timingCacheFile=timing.cache
clear && python3 trt_inf_cupy_resize.py
```