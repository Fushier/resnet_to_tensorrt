import numpy as np
import tensorrt as trt
import cv2
import pycuda.driver as cuda


class HostDeviceMem(object):
    """Simple helper data class that's a little nicer to use than a 2-tuple."""
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

class Infer(object):
    def _load_engine(self, engine_path):
        with open(engine_path, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    
    
    def __init__(self, engine_path: str):
        """Initialize TensorRT plugins, engine and conetxt."""

        self.trt_logger = trt.Logger(trt.Logger.INFO)
        self.engine = self._load_engine(engine_path)

        try:
            self.context = self.engine.create_execution_context()
            self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers(self.engine)
        except Exception as e:
            raise RuntimeError('fail to allocate CUDA resources') from e

    def inference(self, img):
        self.inputs[0].host = np.ascontiguousarray(img)
        trt_outputs = self.do_inference_v2(
            context=self.context,
            bindings=self.bindings,
            inputs=self.inputs,
            outputs=self.outputs,
            stream=self.stream)
        
        return trt_outputs

    
    
    def allocate_buffers(self, engine):
        """Allocates all host/device in/out buffers required for an engine."""
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        return inputs, outputs, bindings, stream
                
    
    def do_inference(self, context, bindings, inputs, outputs, stream, batch_size=1):
        """do_inference (for TensorRT 6.x or lower)
        This function is generalized for multiple inputs/outputs.
        Inputs and outputs are expected to be lists of HostDeviceMem objects.
        """
        # Transfer input data to the GPU.
        [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
        # Run inference.
        context.execute_async(batch_size=batch_size,
                              bindings=bindings,
                              stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
        # Synchronize the stream
        stream.synchronize()
        # Return only the host outputs.
        return [out.host for out in outputs]


    def do_inference_v2(self, context, bindings, inputs, outputs, stream):
        """do_inference_v2 (for TensorRT 7.0+)
        This function is generalized for multiple inputs/outputs for full
        dimension networks.
        Inputs and outputs are expected to be lists of HostDeviceMem objects.
        """
        # Transfer input data to the GPU.
        [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
        # Run inference.
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
        # Synchronize the stream
        stream.synchronize()
        # Return only the host outputs.
        return [out.host for out in outputs]