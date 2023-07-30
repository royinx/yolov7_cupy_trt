import onnx
import sys

model_file = sys.argv[1]
model_name = model_file.split(".onnx")[0]

batch_size = sys.argv[2]
print(f"batch size: {batch_size}")
# assert batch_size.isnumeric()

onnx_output = f'{model_name}_batch_{batch_size}.onnx'

onnx_model = onnx.load(model_file)
print("============== input ==============")
onnx_model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = f'{batch_size}' # only the first input **
print(onnx_model.graph.input[0].name, onnx_model.graph.input[0].type.tensor_type.shape.dim[0].dim_param)
# for idx in range(len(onnx_model.graph.input)):
#     onnx_model.graph.input[idx].type.tensor_type.shape.dim[0].dim_param = f'{batch_size}'
#     print(f"{onnx_model.graph.input[idx].name}, {onnx_model.graph.input[idx].type.tensor_type.shape.dim[0].dim_param}")
print("============== output ==============")
for idx in range(len(onnx_model.graph.output)):
    onnx_model.graph.output[idx].type.tensor_type.shape.dim[0].dim_param = f'{batch_size}'
    print(onnx_model.graph.output[idx].name, onnx_model.graph.output[idx].type.tensor_type.shape.dim[0].dim_param)
    
# print(onnx_model.graph.input[0])
# print(onnx_model.graph.output[0])
onnx.save(onnx_model, onnx_output)

print(f"model saved as {onnx_output}.")
print("Done")

# python3 dynamic.py best.onnx 128