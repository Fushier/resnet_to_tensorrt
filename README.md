# resnet_to_tensorrt
Convert torchvision.models.resnet18 to tensorrt engine manually.
It can be implemented within several codes by torch2trt([NVIDIA-AI-IOT/torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt)):

```
model = torchvision.models.resnet18(pretrained=True)
model = model.eval()
dummy_input = torch.randn((1, 3, 224, 224))
model_trt = torch2trt(model, [dummy_input], strict_type_constraints=True)
```
but writing it manully can help to understand how to use tensorrt.

All the used functions can be founded in [tensorrt docs](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api).



Suggested TensorRT version is 7.0+. The codes are tested with TensorRT 7.0.11 + cuda 10.2 + pytorch 1.8.0 .
