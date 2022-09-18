import io

import cv2
import numpy as np
import onnxruntime as ort
import torch
import yaml
from easydict import EasyDict
from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse

from utils import non_max_suppression, scale_coords, refine_img_size, IMAGENET_MEAN, IMAGENET_STD

# 설정파일 로드
config = yaml.safe_load(open("./config/gmission.proto.yml").read())
config = EasyDict(config)

so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
DETECTION_SESSION = ort.InferenceSession(
    config.detection.model_path,
    providers=['CUDAExecutionProvider'],
    provider_options=[{'device_id': 1}],
    sess_options=so
)

so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
SEGMENTATION_SESSION = ort.InferenceSession(
    config.segmentation.model_path,
    providers=['CUDAExecutionProvider'],
    provider_options=[{'device_id': 1}],
    sess_options=so
)

so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
CLASSIFICATION_SESSION = ort.InferenceSession(
    config.classification.model_path,
    providers=['CUDAExecutionProvider'],
    provider_options=[{'device_id': 1}],
    sess_options=so
)

app = FastAPI()


@torch.no_grad()
@app.post("/detection")
async def get_bboxes_and_diseases(file: UploadFile) -> JSONResponse:
    """
    업로드된 이미지에 대해 넙치 객체의 바운딩 박스, 질병 여부를 반환
    - 바운딩 박스: x1, y1, x2, y2 순서 (Top left & bottom right)
    :param file: 업로드 파일
    :return: JSONResponse
    """

    # TODO. 예외처리
    contents = await file.read()
    img_stream = io.BytesIO(contents)
    img0 = cv2.imdecode(np.frombuffer(img_stream.read(), np.uint8), 1)

    ####################################################################################################################
    # 1. Detection
    ####################################################################################################################

    org_shape = img0.shape[:-1]

    img = cv2.resize(img0, (config.detection.input_size, config.detection.input_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32).transpose(2, 0, 1)[np.newaxis, ...]
    img /= 255.

    output = DETECTION_SESSION.run(None, {'images': img})[0]
    output = torch.tensor(output)
    output = non_max_suppression(output, config.detection.conf_thres, config.detection.iou_thres)[0]
    bboxes = scale_coords(img.shape[2:], output[:, :4], org_shape).round().type(torch.int)

    rois = []
    img = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    for i, (xmin, ymin, xmax, ymax) in enumerate(bboxes):
        roi = img[ymin:ymax + 1, xmin:xmax + 1, :]
        roi = refine_img_size(roi, size=config.segmentation.input_size)
        rois.append(roi)

    rois = np.stack(rois).transpose(0, 3, 1, 2) / 255.

    ####################################################################################################################
    # 2. Segmentation
    ####################################################################################################################
    rois = (rois - np.array(IMAGENET_MEAN)[np.newaxis, :, np.newaxis, np.newaxis]) / np.array(IMAGENET_STD)[np.newaxis,
                                                                                     :, np.newaxis, np.newaxis]
    rois = rois.astype(np.float32)

    seg_map = SEGMENTATION_SESSION.run(None, {'images': rois})[0] > 0

    ####################################################################################################################
    # 2. Classification
    ####################################################################################################################

    labels = CLASSIFICATION_SESSION.run(None, {'images': rois * seg_map})[0].flatten() > 0

    # 응답 반환
    response = dict(
        bboxes=bboxes.tolist(),
        diseases=labels.tolist()
    )

    return JSONResponse(content=response)
