import onnx
import torch
import logging
from torch import nn
import onnxruntime as ort
from PIL import Image
import numpy as np
import cv2
from torchvision.models import resnet50
import torch.nn.functional as F


LOGGER = logging.getLogger()


def export_onnx(model: nn.Module, im: torch.Tensor, output_path: str, opset: int = 12, train: bool = False, simplify: bool = False):
    """Export PyTorch model to Onnx (from YoloV5 repo)"""

    LOGGER.info(f'\nONNX: starting export with onnx {onnx.__version__}...')

    model.eval()

    torch.onnx.export(
        model.cpu(),  # --dynamic only compatible with cpu
        im.cpu(),
        output_path,
        verbose=False,
        opset_version=opset,
        training=torch.onnx.TrainingMode.TRAINING if train else torch.onnx.TrainingMode.EVAL,
        do_constant_folding=not train,
        input_names=['images'],
        output_names=['output'],
        dynamic_axes={
            'images': {0: 'batch'},
            'output': {0: 'batch'}
        })

    # Checks
    model_onnx = onnx.load(output_path)  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model

    # Simplify
    if simplify:
        try:
            cuda = torch.cuda.is_available()
            import onnxsim

            LOGGER.info(f'ONNX: simplifying with onnx-simplifier {onnxsim.__version__}...')
            model_onnx, check = onnxsim.simplify(model_onnx)
            assert check, 'assert check failed'
            onnx.save(model_onnx, output_path)
        except Exception as e:
            LOGGER.info(f'ONNX: simplifier failure: {e}')
    return output_path, model_onnx


if __name__ == "__main__":

    # 네트워크 불러오기
    # state_dict = torch.load("../resources/models/vit.pth", map_location="cpu")['net']
    # model = torch.hub.load("pytorch/vision", "vit_l_16", pretrained=False)
    #
    # model.heads.head = torch.nn.Linear(1024, 1)
    # model.load_state_dict(state_dict)

    img = torch.normal(0, 1, (1, 3, 224, 224))

    # export_onnx(model, img, output_path="../resources/models/vit.latest.onnx", simplify=True)

    # img = cv2.imread("../resources/seg-sample/20220818_108GOPRO_G0403897-4.jpg")
    # img = cv2.resize(img, (128, 128))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
    # img /= 255.

    session = ort.InferenceSession(
        "../resources/models/vit.latest.onnx",
        providers=['CPUExecutionProvider'],
    )
    outputs = session.run(
        None,
        {"images": img.numpy()},
    )

    print(outputs)

    # mask = (outputs[0].squeeze() > 0.5).astype(np.uint8) * 255
    # cv2.imwrite("mask.sample.jpg", mask)

    # print(outputs[0])
