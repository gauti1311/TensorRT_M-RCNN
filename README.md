# Guide to Create A TensorRT Engine from Pretrained Pytorch Model

## Installation : 
```sh
pip install torch torchvision onnx onnxsimplfier  
```
### Install [tesnsorRT](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html)

## Step1 : Convert pytorch model into [ONNX](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html):  
```sh
# Input to the model
x = torch.randn(batch_size, 1, 224, 224, requires_grad=True)
torch_out = torch_model(x)

# Export the model
torch.onnx.export(torch_model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "super_resolution.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})
```
## Step2 : Check the ONNX model and validate output
```sh
  import onnx, onnxruntime
  onnx_model = onnx.load(ONNX_FILE_PATH)
  ## check that the model converted fine
  onnx.checker.check_model(onnx_model)
  print(onnx.helper.printable_graph(onnx_model.graph))
  ort_session = onnxruntime.InferenceSession(ONNX_FILE_PATH, None)
  def to_numpy(tensor):
      return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
  sess_input = ort_session.get_inputs()
  sess_output = ort_session.get_outputs()
  output_names = [ output.name for output in sess_output]
```
## Step3 : Create TensorRT Engine using ONNX model as given [here](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#network_python)
### Minimal Example 
```sh
 import tensorrt as trt
 TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
 
 def build_engine(onnx_file_path):
    explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    builder = trt.Builder(TRT_LOGGER)   # initialize TensorRT engine and parse ONNX model
    network = builder.create_network(explicit_batch)
    parser = trt.OnnxParser(network, TRT_LOGGER)
    builder.max_workspace_size = 1 << 30    # allow TensorRT to use up to 1GB of GPU memory for tactic selection
    builder.max_batch_size = 1  # number of images in Batch
    if builder.platform_has_fast_fp16:  # To use FP16 mode 
        builder.fp16_mode = True
    with open(onnx_file_path, 'rb') as model:    # parse ONNX
        if not parser.parse(model.read()):
            print(parser.get_error(0))
    print('Completed parsing of ONNX file')
    last_layer = network.get_layer(network.num_layers - 1) # if output not detected mark output layer
    network.mark_output(last_layer.get_output(0)) 
    print("total layers", network.num_layers)
    engine = builder.build_cuda_engine(network)   # generate TensorRT engine optimized for the target platform
    context = engine.create_execution_context()
    print("Completed creating Engine")
    return engine, context
    
def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding))
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):  
            input_shape = engine.get_binding_shape(binding)
            inputs.append((host_mem, device_mem))
        else: 
            output_shape = engine.get_binding_shape(binding)
            outputs.append((host_mem, device_mem))
    return inputs, outputs, bindings, stream 

def inference(context, bindings, inputs, outputs, stream, batch_size=1):
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]       # Transfer input data to the GPU.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)     # Run inference.\
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]      # Transfer predictions back from the GPU.
    stream.synchronize()
    return [out.host for out in outputs]       # Return only the host outputs.
    
if __name__ == '__main__':
  engine, context = build_engine(ONNX_FILE_PATH)
  inputs, outputs, bindings, stream = allocate_buffers(engine)
  output = inference(context, bindings, inputs, outputs, stream)
```

#### Notes:  
- The method is tried for simple model and is working
- For trained Sweet Pepper Mask-RCNN ONNX model (converted from torch model) all layers of model cannot be parsed due to some unspportive operations
- ONNX model is required to simplify before to create TensorRT Engine using [ONNX simplifier](https://github.com/daquexian/onnx-simplifier)




