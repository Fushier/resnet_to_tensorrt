# resnet_to_tensorrt
Convert torchvision.models.resnet18 to tensorrt engine manually.
It can be implemented within several codes by torch2trt(NVIDIA-AI-IOT/torch2trt):
```
model = torchvision.models.resnet18(pretrained=True)
model = model.eval()
dummy_input = torch.randn((1, 3, 224, 224))
model_trt = torch2trt(model, [dummy_input], strict_type_constraints=True)
```
but writing it manully can help to understand how to use tensorrt.
