from ultralytics import YOLO
from torchinfo import summary
from thop import profile
import torch


def check_flops(model):
    """check flops and params count"""
    x = torch.zeros(1, 3, 1024, 1024)
    MACs, params = profile(model.model, inputs=(x,))
    print(f"MACs: {MACs / 1e9:.2f} --> GFLOPs: {MACs*2 / 1e9:.2f}")
    print(f"Params: {params / 1e6:.2f} M")

def component(model):
    """Test Module __init__() method"""
    for i, layer in enumerate(model.model.model):
        print(f"==Layer {i}: {layer}\n")

def output_shape(model):
    """Test Module forward() method. Note: don't trust the parameters count by this."""
    summary(model.model, input_size=(1, 3, 640, 640))  # 640 follows the paper


if __name__ == "__main__":
    model = YOLO("model/LiteUAV_Det_m.yaml", task="detect")
    model.info(imgsz=1024)

    # check_flops(model)
    # component(model)
    output_shape(model)
