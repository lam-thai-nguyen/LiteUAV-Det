from ultralytics import YOLO
from torchinfo import summary
from thop import profile
import torch


def check_flops(model):
    """should print non-zero FLOPs if working"""
    x = torch.zeros(1, 3, 640, 640)
    MACs, _ = profile(model.model, inputs=(x,))
    print(f"MACs: {MACs / 1e9:.2f} --> GFLOPs: {MACs*2 / 1e9:.2f}")

def component(model):
    """Test Module __init__() method"""
    for i, layer in enumerate(model.model.model):
        print(f"==Layer {i}: {layer}\n")

def output_shape(model):
    """Test Module forward() method"""
    summary(model.model, input_size=(1, 3, 640, 640))


if __name__ == "__main__":
    model = YOLO("model/LiteUAV_Det_l.yaml", task="detect")

    # check_flops(model)
    # component(model)
    output_shape(model)
