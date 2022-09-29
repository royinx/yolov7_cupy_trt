import time 
import sys
import os

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import cv2

# from turbojpeg import TurboJPEG
from nvjpeg import NvJpeg 
from line_profiler import LineProfiler

# jpeg = TurboJPEG()
nj = NvJpeg()
profile = LineProfiler()

# add EfficientNMS_TRT plugin
trt.init_libnvinfer_plugins(None,"")

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

class TensorRT(object):
# class TensorRT(RootClass):
    def __init__(self,engine_file):
        super().__init__()
        self.TRT_LOGGER = trt.Logger()
        self.engine = self.get_engine(engine_file)
        self.context = self.engine.create_execution_context()
        self.max_batch_size = self.engine.max_batch_size
        self.allocate_buffers()

    def get_engine(self, engine_file_path):
        if os.path.exists(engine_file_path):
            print("Reading engine from file {}".format(engine_file_path))
            with open(engine_file_path, "rb") as f, \
                trt.Runtime(self.TRT_LOGGER) as runtime:
                return runtime.deserialize_cuda_engine(f.read())
        raise "Cannot find .trt engine file."

    def allocate_buffers(self):
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()

        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            self.bindings.append(int(device_mem))
            # print(bindings)
            # Append to the appropriate list.
            if self.engine.binding_is_input(binding):
                self.inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                self.outputs.append(HostDeviceMem(host_mem, device_mem))

        # return inputs, outputs, bindings, stream
    @profile
    def inference(self,inputs:np.ndarray) -> list: # input: <NCHW>
        self.inputs[0].host = inputs
        [cuda.memcpy_htod_async(inp.device, inp.host, self.stream) for inp in self.inputs]
        self.context.execute_async_v2(bindings=self.bindings,
                                    stream_handle=self.stream.handle)
        [cuda.memcpy_dtoh_async(out.host, out.device, self.stream) for out in self.outputs]
        self.stream.synchronize()
        return [out.host for out in self.outputs]

class YoloTRT(object):
    def __init__(self):
        super().__init__()
        self.max_batch_size = 64
        self.trt = TensorRT(f"/py/ped_yolov7_{self.max_batch_size}.trt")
        
    def _preprocess(self, input_array:np.ndarray) -> np.ndarray: # 
        rescale_info = []
        
        
        output_array = np.zeros((640,640,3),input_array.dtype) # NHWC
        resize_scale, top_pad, left_pad, out_img = resize_image(input_array[0], output_array)
        # print(out_img)
        # cv2.imwrite("test3.jpg",out_img[:,:,::-1])
        rescale_info.append([resize_scale, top_pad, left_pad])
        output_array = np.tile(out_img,[self.max_batch_size,1,1,1]) # NHWC
        output_array = np.transpose(output_array,(0,3,1,2)) # NHWC -> NCHW
        output_array = np.ascontiguousarray(output_array)
        output_array = output_array.astype(np.float32)
        output_array/=255

        output_array = np.ascontiguousarray(output_array)
        return output_array, rescale_info # in: <NHWC> raw image batch , out: <NCHW> resized <N,3,608,608>

    def _inference(self, input: np.ndarray) -> list:
        trt_outputs = self.trt.inference(input)
        output_shapes = [(self.max_batch_size, 1),
                         (self.max_batch_size, 100,4),
                         (self.max_batch_size, 100),
                         (self.max_batch_size, 100)]
                         
        trt_outputs = [output.reshape(shape) for output, shape in zip(trt_outputs, output_shapes)]  # [(N, 1), (N, 100, 4), (N, 100), (N, 100)]
        return trt_outputs # in: <NCHW> <N,3,640,640>, out: [(N, 1), (N, 100, 4), (N, 100), (N, 100)]

    def _postprocess(self, feat_batch, rescale_info) -> list:
        def scale_coord(boxes,resize_scale,top_pad,left_pad):
            boxes[:, [0, 2]] -= left_pad  # x padding
            boxes[:, [1, 3]] -= top_pad  # y padding
            boxes[:, :4] /= resize_scale
            # dwdh = np.array(dwdh*2)
            # boxes -= dwdh
            # boxes /= r
            return boxes

        rs = []
        for feat, (resize_scale, top_pad, left_pad )in zip(feat_batch, rescale_info):
            num_dets, det_boxes, det_scores, det_classes = feat
            boxes = scale_coord(det_boxes[:num_dets[0]],resize_scale,top_pad,left_pad).round().astype(np.int)
            scores = det_scores[:num_dets[0]]
            classes = det_classes[:num_dets[0]]
            rs.append([boxes, scores, classes])
        
        return rs

    @profile
    def inference(self, input_array:np.ndarray): # img_array <N,H,W,C>

        pre, rescale_info = self._preprocess(input_array) # in: <NHWC> raw image batch , out: <NCHW> resized <N,3,608,608>
        trt_outputs = self._inference(pre) # out: [(N, 1), (N, 100, 4), (N, 100), (N, 100)]

        feat_batch = [[trt_outputs[j][i] for j in range(len(trt_outputs))] for i in range(len(trt_outputs[0]))]
        # print(len(feat_batch))
        # for i in feat_batch[0]:
        #     print(i.shape)
        # print(len(feat_batch[0]))

        # exit()
        post = self._postprocess(feat_batch, rescale_info)

        print(post)

        
        imgs = np.tile(input_array,[self.max_batch_size,1,1,1]) # NHWC
        print(imgs.shape)
        for img , (boxes,scores,classes) in zip(imgs,post):
            for box, score, cl in zip(boxes,scores,classes):
                name = str(cl)
                color = (255,0,0)
                name += ' ' + str(round(float(score),3))
                cv2.rectangle(img,tuple(box[:2].tolist()),tuple(box[2:].tolist()),color,2)
                cv2.putText(img,name,(int(box[0]), int(box[1]) - 2),cv2.FONT_HERSHEY_SIMPLEX,0.75,color,thickness=2)
            cv2.imwrite("test4.jpg",img[:,:,::-1])

        # post = post[:len(input_array)]
        return post


def resize_image(img: np.ndarray, out_img: np.ndarray) -> (float, int, int):
    assert img.dtype == out_img.dtype, "Input images must have same dtype"
    left_pad = 0
    top_pad = 0
    h, w, _ = out_img.shape
    if img.shape[0] / img.shape[1] > h / w:
        resize_scale = h / img.shape[0]
        tmp_img = cv2.resize(img, (0, 0), fx=resize_scale, fy=resize_scale)
        left_pad = int((w - tmp_img.shape[1]) / 2)
        out_img[:, left_pad:left_pad + tmp_img.shape[1], :] = tmp_img
    else:
        resize_scale = w / img.shape[1]
        tmp_img = cv2.resize(img, (0, 0), fx=resize_scale, fy=resize_scale)
        top_pad = int((h - tmp_img.shape[0]) / 2)
        out_img[top_pad:top_pad + tmp_img.shape[0], :, :] = tmp_img
    return resize_scale, top_pad, left_pad, out_img


def unit_test():
    input_image_path = '/py/crop/9/3.jpg'
    with open(input_image_path, 'rb') as infile:
        image_raw = nj.decode(infile.read())
        image_raw = image_raw[:,:,::-1] # to RGB
    image_raw = np.tile(image_raw,[1,1,1,1])

    yolo = YoloTRT()
    batch_size = yolo.trt.max_batch_size
    for i in range(0, len(image_raw), batch_size):
        batch = image_raw[i:i+batch_size]
        rs = yolo.inference(batch)

    profile.print_stats()


def main():
    unit_test()

if __name__ == '__main__':
    main()


# python /py/yolov7/export.py --weights /py/best.pt --batch-size 8 --grid --end2end --simplify --topk-all 100 --iou-thres 0.45 --conf-thres 0.25 --img-size 640 640
# mv /py/best.onnx /py/ped_yolov7_8.onnx
# trtexec --onnx=/py/ped_yolov7_64.onnx --saveEngine=/py/ped_yolov7_64.trt  --fp16 --workspace=8192 --explicitBatch --dumpOutput --timingCacheFile=timing.cache
# clear && python3 trt_inf.py
