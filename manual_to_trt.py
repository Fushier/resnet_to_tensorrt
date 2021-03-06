import os
import sys
import time
import torchvision
import torch
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
from PIL import Image
from torchvision import transforms
from data import TestImageDataset
from infer import Infer
from progressbar import progressbar


# You can set the logger severity higher to suppress messages (or lower to display more messages).
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
data_dir = 'test_images'
batch_size = 1
engine_path = 'resnet18.trt'

class ModelData(object):
    LABEL_PATH = "labels.txt"
    INPUT_NAME = "data"
    INPUT_SHAPE = (1, 3, 224, 224)
    OUTPUT_NAME = "class"
    OUTPUT_SIZE = 1000
    DTYPE = trt.float32


# TensorRT does not provide the function of BN layer, but only provides the most basic scale layer 
# It needs to be implemented manually according to BN formula
def add_bn_layer(network, last_layer, weights, bn_name: str):
    gamma = weights[bn_name + r'.weight'].numpy()
    beta = weights[bn_name + r'.bias'].numpy()
    mean = weights[bn_name + r'.running_mean'].numpy()
    var = weights[bn_name + r'.running_var'].numpy()
    scale = gamma / np.sqrt(var + 1e-05)
    bias = beta - mean * scale
    power = np.ones_like(scale)
    
    return network.add_scale(last_layer.get_output(0), trt.ScaleMode.CHANNEL, bias, scale, power)

# bias is set False in Conv2d, seen in print(model)
def add_conv_layer(network, last_layer, weights, conv_name: str, out_put_maps, kernel_shape, stride, padding):
    weight = weights[conv_name + r'.weight'].numpy()
    conv = network.add_convolution(last_layer.get_output(0), out_put_maps, kernel_shape, weight)
    conv.stride = stride
    conv.padding = padding
    
    return conv
    
def add_downsample(network, last_layer, weights, down_name: str, out_put_maps):
    downsample_conv = add_conv_layer(network, last_layer, weights, down_name + r'.0', out_put_maps, (1, 1), (2, 2), (0, 0))
    return add_bn_layer(network, downsample_conv, weights, down_name + r'.1')


def basic_block(network, weights, last_layer, inch: int, outch: int, stride: int, layer_name: str):
    conv1 = add_conv_layer(network, last_layer, weights, layer_name + 'conv1', outch, (3, 3), (stride, stride), (1, 1))
    bn1 = add_bn_layer(network, conv1, weights, layer_name + 'bn1')
    relu1 = network.add_activation(bn1.get_output(0), type=trt.ActivationType.RELU)
    
    conv2 = add_conv_layer(network, relu1, weights, layer_name + 'conv2', outch, (3, 3), (1, 1), (1, 1))
    bn2 = add_bn_layer(network, conv2, weights, layer_name + 'bn2')
    if inch != outch:
        bn3 = add_downsample(network, last_layer, weights, layer_name + 'downsample', outch)
        ew = network.add_elementwise(bn3.get_output(0), bn2.get_output(0), trt.ElementWiseOperation.SUM)
    else:
        ew = network.add_elementwise(last_layer.get_output(0), bn2.get_output(0), trt.ElementWiseOperation.SUM)
    
    relu2 = network.add_activation(ew.get_output(0), type=trt.ActivationType.RELU)
    return relu2
   
# see all the weights name in model.state_dict().keys()
def populate_network(network, weights):
    # Configure the network layers based on the weights provided.
    input_tensor = network.add_input(name=ModelData.INPUT_NAME, dtype=ModelData.DTYPE, shape=ModelData.INPUT_SHAPE)

    conv1_w = weights['conv1.weight'].numpy()
    conv1 = network.add_convolution(input=input_tensor, num_output_maps=64, kernel_shape=(7, 7), kernel=conv1_w)
    conv1.stride = (2, 2)
    conv1.padding = (3, 3)
    

    bn1 = add_bn_layer(network, conv1, weights, 'bn1')

    relu1 = network.add_activation(input=bn1.get_output(0), type=trt.ActivationType.RELU)

    maxpool = network.add_pooling(input=relu1.get_output(0), type=trt.PoolingType.MAX, window_size=(3, 3))
    maxpool.stride = (2, 2)
    maxpool.padding = (1, 1)
    
    relu2 = basic_block(network, weights, maxpool, 64, 64, 1, 'layer1.0.')
    relu3 = basic_block(network, weights, relu2, 64, 64, 1, 'layer1.1.')
    
    relu4 = basic_block(network, weights, relu3, 64, 128, 2, 'layer2.0.')
    relu5 = basic_block(network, weights, relu4, 128, 128, 1, 'layer2.1.')
    
    relu6 = basic_block(network, weights, relu5, 128, 256, 2, 'layer3.0.')
    relu7 = basic_block(network, weights, relu6, 256, 256, 1, 'layer3.1.')
    
    relu8 = basic_block(network, weights, relu7, 256, 512, 2, 'layer4.0.')
    relu9 = basic_block(network, weights, relu8, 512, 512, 1, 'layer4.1.')
    
    # alternative way to implement pytorch adaptiveAvgPool2d
    output_size = (1, 1)
    avg_input_tensor = relu9.get_output(0)
    stride = (avg_input_tensor.shape[-2] // output_size[-2], avg_input_tensor.shape[-1] // output_size[-1])

    kernel_size = stride
    adaptive_avg_Pool = network.add_pooling(input=avg_input_tensor, type=trt.PoolingType.AVERAGE, window_size=kernel_size)
    adaptive_avg_Pool.stride = stride
    
    
    fc_w = weights['fc.weight'].numpy()
    fc_b = weights['fc.bias'].numpy()
    fc = network.add_fully_connected(adaptive_avg_Pool.get_output(0), ModelData.OUTPUT_SIZE, fc_w, fc_b)

    fc.get_output(0).name = ModelData.OUTPUT_NAME
    network.mark_output(tensor=fc.get_output(0))

def build_engine(weights):
    # For more information on TRT basics, refer to the introductory samples.
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(EXPLICIT_BATCH)
    config = builder.create_builder_config()
    runtime = trt.Runtime(TRT_LOGGER)

    config.max_workspace_size = 1 << 30
    config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
    # uncomment this line to use FP16 mode
    # config.set_flag(trt.BuilderFlag.FP16)
    # Populate the network using weights from the PyTorch model.
    populate_network(network, weights)
    # Build and return an engine.
    return builder.build_engine(network, config)

def main():
    model = torchvision.models.resnet18(pretrained=True)
    labels = open(ModelData.LABEL_PATH, 'r').read().split('\n')

    weights = model.state_dict()
    if os.path.exists(engine_path) == False:
        # Do inference with TensorRT.
        print('trt engine not found, start populating network')
        engine = build_engine(weights)
        print('engine serializing...')
        with open(engine_path, 'wb') as f:
            f.write(engine.serialize())
        print('engine serialize finished')

    model = model.eval().to(device)
    infer_obj = Infer(engine_path)
    dataset = TestImageDataset(data_dir)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    torch_preds = list()
    trt_preds = list()
    print('inference with TensorRT...')
    t1 = time.time()
    for data in progressbar(dataloader):
        trt_output = infer_obj.inference(data.numpy())
        trt_pred = np.argmax(trt_output[0])
        trt_preds.append(trt_pred)
    t2 = time.time()
    print('inference with PyTorch...')
    for data in progressbar(dataloader):
        resnet_input = data.to(device)
        resnet_output = model(resnet_input)
        _, preds = torch.max(resnet_output, 1)
        preds = preds.cpu().numpy().tolist()
        torch_preds += preds
    t3 = time.time()

    print("TensorRT inference time:", t2 - t1, "s")
    print("PyTorch inference time:", t3 - t2, "s")
    correct = 0
    for i in range(len(trt_preds)):
        if trt_preds[i] == torch_preds[i]:
            correct += 1
    print("acc:", correct / len(trt_preds))

if __name__ == '__main__':
    main()
    
