# Yolov7 inference pipeline
### Pros
Ultra High speed kernel in GPU using:
 - NvJpeg
 - cupy
 - CUDA Module preprocessing ( Resize + Padding + Transpose + Normalise )
 - TensorRT


### Torch -> ONNX
```bash
# Setup Env
docker build -t yolov7_onnx -f dockerfile_onnx .
docker run --rm \
           --runtime=nvidia \
           --name=yolov7_onnx \
           --privileged \
           --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
           -it \
           -v $PWD:/py/ \
           -w /py \
           yolov7_onnx bash
python3 /py/export.py --weights /py/best.pt --batch-size 64 --grid --end2end --simplify --topk-all 100 --iou-thres 0.45 --conf-thres 0.25 --img-size 640 640
mv /py/best.onnx /py/ped_yolov7.onnx
```

### ONNX -> TensorRT

```bash
# Setup Env
docker build -t yolov7_trt -f dockerfile_trt .
docker run --rm \
           --runtime=nvidia \
           --name=yolov7_trt \
           --privileged \
           -it \
           -v $PWD:/py/ \
           -w /py \
           yolov7_trt bash

# git clone https://github.com/WongKinYiu/yolov7
# git clone https://github.com/Monday-Leo/YOLOv7_Tensorrt.git
```

<details open><summary> INT8 </summary>

#### Reference

- Backbone script : https://github.com/qq995431104/Pytorch2TensorRT
- TensorRT 8 cmd : https://github.com/NVIDIA/TensorRT/blob/release/8.6/samples/python/detectron2/build_engine.py
- Quantization   : https://github.com/xuanandsix/Tensorrt-int8-quantization-pipline

```bash
# install coco and download image for calibration
cd int8
git clone https://github.com/waleedka/coco
pip3 install cython matplotlib requests tqdm
cd coco/PythonAPI && make && make install && python3 setup.py install && cd - 
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip && unzip annotations_trainval2017.zip
python3 download.py
mv downloaded_images Pytorch2TensorRT
```
```bash
cd Pytorch2TensorRT && pip3 install torchvision
python3 main.py --batch_size 32 \
                --height 640 \
                --width 640 \
                --cache_file ./cal.cache \
                --mode int8 \
                --onnx_file_path /py/ped_yolov7.onnx \
                --engine_file_path /py/ped_yolov7_int8.trt
```
</details>

<details><summary> FP16 </summary>

```bash
trtexec --onnx=/py/ped_yolov7.onnx \
        --saveEngine=/py/ped_yolov7_fp16.trt \
        --fp16 --explicitBatch --dumpOutput \
        --memPoolSize=workspace:8192

```
</details>

### Inference

```bash
clear && python3 trt_inf_cupy_cpu_resize.py
```