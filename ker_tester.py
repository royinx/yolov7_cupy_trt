import cv2
import cupy as cp
from nvjpeg import NvJpeg 
from line_profiler import LineProfiler
from _module import SourceModule

nj = NvJpeg()
profile = LineProfiler()

module = cp.RawModule(code=SourceModule)

cuResizeKer = module.get_function("cuResize")
TransposeKer = module.get_function("Transpose")
TransNorKer = module.get_function("Transpose_and_normalise")


loaded_from_source = r'''
extern "C"{

__global__ void test_sum(const float* x1, const float* x2, float* y, \
                         unsigned int N)
{
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < N)
    {
        y[tid] = x1[tid] + x2[tid];
    }
}

__global__ void test_multiply(const float* x1, const float* x2, float* y, \
                              unsigned int N)
{
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < N)
    {
        y[tid] = x1[tid] * x2[tid];
    }
}

}'''
module = cp.RawModule(code=loaded_from_source)
ker_sum = module.get_function('test_sum')
ker_times = module.get_function('test_multiply')



N = 10
x1 = cp.arange(N**2, dtype=cp.float32).reshape(N, N)
x2 = cp.ones((N, N), dtype=cp.float32)
y = cp.zeros((N, N), dtype=cp.float32)
ker_sum((N,), (N,), (x1, x2, y, N**2))   # y = x1 + x2
assert cp.allclose(y, x1 + x2)
ker_times((N,), (N,), (x1, x2, y, N**2)) # y = x1 * x2
assert cp.allclose(y, x1 * x2)




batch = 50

# src_h = 1080
# src_w = 1920
# inp_batch = cp.ones((batch, src_h, src_w, 3), dtype=cp.uint8)

img = cp.array(nj.read("4k.jpg"))
src_h, src_w, _ = img.shape
inp_batch = cp.tile(img, [batch,1,1,1])
dst_h = 640
dst_w = 640
out_batch = cp.zeros((batch, dst_h, dst_w, 3), dtype=cp.uint8)

# block = (32, 32, 1)   # blockDim | threadIdx 
# grid = (19,19,3)     # gridDim  | blockIdx

DST_SIZE = dst_h * dst_w * 3
block = (1024, )
grid = (int(DST_SIZE/3//1024)+1,batch,3)


for _ in range(10): # warm up
    cuResizeKer(grid, block, 
                (inp_batch, out_batch, 
                cp.int32(src_h), cp.int32(src_w),
                cp.int32(dst_h), cp.int32(dst_w),
                cp.float32(src_h/dst_h), cp.float32(src_w/dst_w))
                )

@profile
def inner():
    cuResizeKer(grid, block, 
                (inp_batch, out_batch, 
                cp.int32(src_h), cp.int32(src_w),
                cp.int32(dst_h), cp.int32(dst_w),
                cp.float32(src_h/dst_h), cp.float32(src_w/dst_w))
                )

for _ in range(100):
    inner()

# out_batch
# print(out_batch)

host_img  = cp.asnumpy(out_batch)[0]
print(host_img.shape)
cv2.imwrite("test_resize.jpg", host_img)


profile.print_stats()



# clear && python3 ker_tester.py
