import os

import numpy as np
import cupy as cp
import tensorrt as trt
import cv2

from nvjpeg import NvJpeg
from line_profiler import LineProfiler

nj = NvJpeg()
profile = LineProfiler()

# add EfficientNMS_TRT plugin
trt.init_libnvinfer_plugins(None,"")

pinned_memory_pool = cp.cuda.PinnedMemoryPool()
cp.cuda.set_pinned_memory_allocator(pinned_memory_pool.malloc)

with open('lib_cuResize.cu', 'r', encoding="utf-8") as reader:
    module = cp.RawModule(code=reader.read())

cuResizeKer = module.get_function("cuResize")

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

class TensorRT(object):
    def __init__(self,engine_file):
        super().__init__()
        self.TRT_LOGGER = trt.Logger()
        self.engine = self.get_engine(engine_file)
        self.context = self.engine.create_execution_context()
        # self.max_batch_size = self.engine.max_batch_size # must be 1 if explicit batch
        self.allocate_buffers()

    def get_engine(self, engine_file_path):
        if os.path.exists(engine_file_path):
            print("Reading engine from file {}".format(engine_file_path))
            with open(engine_file_path, "rb") as f, \
                trt.Runtime(self.TRT_LOGGER) as runtime:
                return runtime.deserialize_cuda_engine(f.read())
        raise "Cannot find .trt engine file."

    def allocate_buffers(self):
        """
        In this Application, we use cupy for in and out

        trt use gpu array to run inference.
        while bindings store the gpu array ptr , via the method : int(cupy.ndarray.data) / cp.cuda.alloc_pinned_memory / cuda.mem_alloc

        So HostDeviceMem is not hard requirement but a easier way to pin the memory and copy back to host.
        """
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cp.cuda.Stream(non_blocking=False)

        for binding in self.engine:
            shape = self.engine.get_tensor_shape(binding)
            dtype = trt.nptype(self.engine.get_tensor_dtype(binding))
            device_array = cp.empty(shape, dtype)
            self.bindings.append(int(device_array.data)) # cupy array ptr
            # Append to the appropriate list.
            if self.engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
                self.inputs.append(HostDeviceMem(None, device_array))
            elif self.engine.get_tensor_mode(binding) == trt.TensorIOMode.OUTPUT:
                self.outputs.append(HostDeviceMem(None, device_array))

    @profile
    def inference(self, inputs:[cp.ndarray, np.ndarray]) -> list: # input: <NCHW>
        N = inputs.shape[0]
        if inputs.dtype != self.inputs[0].device.dtype:
            inputs = inputs.astype(self.inputs[0].device.dtype)
        if isinstance(inputs, cp.ndarray): # DtoD
            inputs = cp.ascontiguousarray(inputs)
            cp.cuda.runtime.memcpy(dst = int(self.inputs[0].device.data), # dst_ptr
                                   src = int(inputs.data), # src_ptr
                                   size=inputs.nbytes,
                                   kind=3) # 0: HtoH, 1: HtoD, 2: DtoH, 3: DtoD, 4: unified virtual addressing
        elif isinstance(inputs, np.ndarray):
            inputs = np.ascontiguousarray(inputs)
            cp.cuda.runtime.memcpy(dst = int(self.inputs[0].device.data), # dst_ptr
                                   src = inputs.ctypes.data, # src_ptr
                                   size=inputs.nbytes,
                                   kind=1)
        # print(self.inputs[0].device)
        # if self._dynamic:
        #     self.context.set_binding_shape(0, inputs.shape)
            # self.context.set_binding_shape(-1, (N,512))
        self.context.execute_async_v2(bindings = self.bindings,
                                      stream_handle = self.stream.ptr)
        self.stream.synchronize()
        return [out.device[:N] for out in self.outputs]

    # @profile
    # def inference(self,inputs:np.ndarray) -> list: # input: <NCHW>
    #     # print(self.inputs[0].device.shape , inputs.shape)
        
    #         self.inputs[0].device.set(inputs)

    #         cp.cuda.runtime.memcpy(dst = int(self.inputs[0].device.data), # dst_ptr
    #                                         src = int(inputs.data), # src_ptr
    #                                         size=inputs.nbytes,
    #                                         kind=3)

    #     self.context.execute_async_v2(bindings=self.bindings,
    #                                 stream_handle=self.stream.ptr)
    #     self.stream.synchronize()
    #     return [out.device for out in self.outputs]

class YoloTRT(object):
    def __init__(self):
        super().__init__()
        # self.trt = TensorRT(f"/py/ped_yolov7_{self.max_batch_size}.trt")
        self.trt = TensorRT(f"/py/ped_yolov7_fp16.trt")
        # self.trt = TensorRT(f"/py/ped_yolov7_int8.trt")
        self.max_batch_size = 64

    def _preprocess(self, input_array) -> cp.ndarray: #
        # print(input_array.shape)
        input_array_gpu = cp.empty(shape=input_array.shape, dtype=input_array.dtype)

        if isinstance(input_array, cp.ndarray): # DtoD
            cp.cuda.runtime.memcpy(dst = int(input_array_gpu.data), # dst_ptr
                                    src = int(input_array.data), # src_ptr
                                    size=input_array.nbytes,
                                    kind=3) # 0: HtoH, 1: HtoD, 2: DtoH, 3: DtoD, 4: unified virtual addressing
        elif isinstance(input_array, np.ndarray):
            cp.cuda.runtime.memcpy(dst = int(input_array_gpu.data), # dst_ptr
                                    src = input_array.ctypes.data, # src_ptr
                                    size=input_array.nbytes,
                                    kind=1)

        # output_array = np.zeros((640,640,3),input_array.dtype) # NHWC
        resize_shape = (640,640)
        # batch_array = np.array(input_array)
        resize_scale, top_pad, left_pad, output_array = cuda_resize(input_array_gpu, resize_shape, pad = False)
        rescale_info = [[resize_scale, top_pad, left_pad]for _ in range(len(input_array))]
        
        # CPU resize
            # resize_scale, top_pad, left_pad, out_img = resize_image(input_array[0], output_array)
            # print(out_img)
            # cv2.imwrite("test3.jpg",out_img[:,:,::-1])
            # output_array = np.tile(out_img,[self.max_batch_size,1,1,1]) # NHWC
            # output_array = np.transpose(output_array,(0,3,1,2)) # NHWC -> NCHW
            # output_array = np.ascontiguousarray(output_array)
            # output_array = output_array.astype(np.float32)
            # output_array/=255
            # output_array = np.ascontiguousarray(output_array)
        # rescale_info.append([resize_scale, top_pad, left_pad])
        return output_array, rescale_info # in: <NHWC> raw image batch , out: <NCHW> resized <N,3,608,608>

    def _inference(self, input: np.ndarray) -> list:
        trt_outputs = self.trt.inference(input)
        return trt_outputs # in: <NCHW> <N,3,640,640>, out: [(N, 1), (N, 100, 4), (N, 100), (N, 100)]

    def _postprocess(self, feat_batch, rescale_info) -> list:
        def scale_coord(boxes,resize_scale,top_pad,left_pad):
            boxes[:, [0, 2]] -= left_pad  # x padding
            boxes[:, [1, 3]] -= top_pad  # y padding
            boxes[:, :4] /= resize_scale
            return boxes

        rs = []
        for feat, (resize_scale, top_pad, left_pad )in zip(feat_batch, rescale_info):
            num_dets, det_boxes, det_scores, det_classes = feat
            boxes = scale_coord(det_boxes[:num_dets[0]],resize_scale,top_pad,left_pad).round().astype(int)
            scores = det_scores[:num_dets[0]]
            classes = det_classes[:num_dets[0]]
            rs.append([boxes, scores, classes])

        return rs

    @profile
    def inference(self, input_array:np.ndarray): # img_array <N,H,W,C>

        pre, rescale_info = self._preprocess(input_array) # in: <NHWC> raw image batch , out: <NCHW> resized <N,3,608,608>
        trt_outputs = self._inference(pre) # out: [(N, 1), (N, 100, 4), (N, 100), (N, 100)]
        feat_batch = [[trt_outputs[j][i] for j in range(len(trt_outputs))] for i in range(len(trt_outputs[0]))]
        post = self._postprocess(feat_batch, rescale_info)

        return post

def cuda_resize(inputs: cp.ndarray, # src: (N,H,W,C)
                shape: tuple, # (dst_h, dst_w)
                out: cp.ndarray=None, # dst: (N,H,W,C)
                pad: bool=True):

    out_dtype = cp.uint8

    N, src_h, src_w, C = inputs.shape
    assert C == 3 # resize kernel only accept 3 channel tensors.
    dst_h, dst_w = shape
    DST_SIZE = dst_h * dst_w * C

    # define kernel configs
    block = (1024, )
    grid = (int(DST_SIZE/3//1024)+1,N,3)

    if len(shape)!=2:
        raise "cuda resize target shape must be (h,w)"
    if out:
        assert out.dtype == out_dtype
        assert out.shape[1] == dst_h
        assert out.shape[2] == dst_w

    resize_scale = 1
    left_pad = 0
    top_pad = 0
    if pad:
        padded_batch = cp.zeros((N, dst_h, dst_w, C), dtype=out_dtype)
        if src_h / src_w > dst_h / dst_w:
            resize_scale = dst_h / src_h
            ker_h = dst_h
            ker_w = int(src_w * resize_scale)
            left_pad = int((dst_w - ker_w) / 2)
        else:
            resize_scale = dst_w / src_w
            ker_h = int(src_h * resize_scale)
            ker_w = dst_w
            top_pad = int((dst_h - ker_h) / 2)
    else:
        ker_h = dst_h
        ker_w = dst_w

    shape = (N, ker_h, ker_w, C)
    if not out:
        out = cp.empty(tuple(shape),dtype = out_dtype)

    with cp.cuda.stream.Stream() as stream:
        cuResizeKer(grid, block,
                (inputs, out,
                cp.int32(src_h), cp.int32(src_w),
                cp.int32(ker_h), cp.int32(ker_w),
                cp.float32(src_h/ker_h), cp.float32(src_w/ker_w)
                )
            )
        if pad:
            if src_h / src_w > dst_h / dst_w:
                padded_batch[:, :, left_pad:left_pad + out.shape[2], :] = out
            else:
                padded_batch[:, top_pad:top_pad + out.shape[1], :, :] = out
            padded_batch = cp.ascontiguousarray(padded_batch)
        stream.synchronize()

    if pad:
        return resize_scale, top_pad, left_pad, padded_batch
    return resize_scale, top_pad, left_pad, out


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
    import os 
    image_path = [os.path.join('/py/test_img/',file_)for file_ in os.listdir('/py/test_img/')]
    
    yolo = YoloTRT()
    batch_size = yolo.max_batch_size

    image_raw = []
    for i in range(batch_size//len(image_path)+1):
        for input_image_path in image_path:
            with open(input_image_path, 'rb') as infile:
                image = nj.decode(infile.read())
                image_raw.append(image[:,:,::-1]) # to RGB

    image_raw = np.asarray(image_raw)

    
    for i in range(0, len(image_raw), batch_size):
        batch = image_raw[i:i+batch_size]
        rs = yolo.inference(batch)
        # exit()
        for img , (boxes,scores,classes) in zip(batch,rs):
            for box, score, cl in zip(boxes,scores,classes):
                name = str(cl)
                color = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))
                name += ' ' + str(round(float(score),3))
                cv2.rectangle(img,tuple(box[:2].tolist()),tuple(box[2:].tolist()),color,1)
                cv2.putText(img,name,(int(box[0]), int(box[1]) - 2),cv2.FONT_HERSHEY_SIMPLEX,0.75,color,thickness=1)
            cv2.imwrite("test4.jpg",img[:,:,::-1])
            import time
            time.sleep(2)
    # profile.print_stats()


def main():
    unit_test()

if __name__ == '__main__':
    main()


# python /py/yolov7/export.py --weights /py/best.pt --batch-size 8 --grid --end2end --simplify --topk-all 100 --iou-thres 0.45 --conf-thres 0.25 --img-size 640 640
# mv /py/best.onnx /py/ped_yolov7_8.onnx
# trtexec --onnx=/py/ped_yolov7_64.onnx --saveEngine=/py/ped_yolov7_64.trt  --fp16 --workspace=8192 --explicitBatch --dumpOutput --timingCacheFile=timing.cache
# clear && python3 trt_inf_cupy_resize.py
