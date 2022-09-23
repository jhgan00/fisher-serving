import io

import cv2
import numpy as np
import torch
import yaml
from easydict import EasyDict
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from utils import non_max_suppression, scale_coords, pad_and_resize, IMAGENET_MEAN, IMAGENET_STD
from inference_session import initialize_session

# 설정파일 로드
config = yaml.safe_load(open("./config/gmission.proto.yml").read())
config = EasyDict(config)

DETECTION_SESSION = initialize_session(config.detection.model_path, config.detection.device_id)
SEGMENTATION_SESSION = initialize_session(config.segmentation.model_path, config.segmentation.device_id)
CLASSIFICATION_SESSION = initialize_session(config.classification.model_path, config.classification.device_id)

app = FastAPI()


# TODO. 키 적용
@torch.no_grad()
@app.post("/detection")
async def get_bboxes_and_diseases(file: UploadFile) -> JSONResponse:
    """
    업로드된 이미지에 대해 넙치 객체의 바운딩 박스, 질병 여부를 반환
    - bboxes: x1, y1, x2, y2 순서 (Top left & bottom right)
    - diseases: 개체에 하나 이상의 질병이 있는 경우 1, 없는 경우 0
    :param file: 업로드 파일
    :return: JSONResponse
    """

    # if not file.content_type.startswith("image"):
    #     raise HTTPException(400, detail=f"Invalid content type: {file.content_type}")

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

    if not len(bboxes):
        response = dict(
            bboxes=[],
            diseases=[],
            num_objects=len(bboxes)
        )
        return JSONResponse(content=response)

    rois = []
    img = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    for i, (xmin, ymin, xmax, ymax) in enumerate(bboxes):
        roi = img[ymin:ymax + 1, xmin:xmax + 1, :]
        roi = pad_and_resize(roi, size=config.segmentation.input_size)
        rois.append(roi)

    rois = np.stack(rois).transpose(0, 3, 1, 2) / 255.

    ####################################################################################################################
    # 2. Segmentation
    ####################################################################################################################

    # standardization
    rois -= IMAGENET_MEAN
    rois /= IMAGENET_STD
    rois = rois.astype(np.float32)
    batch_size = config.segmentation.batch_size
    seg_map = SEGMENTATION_SESSION.batch_run(rois, batch_size)
    seg_map = seg_map > config.segmentation.threshold

    ####################################################################################################################
    # 3. Classification
    ####################################################################################################################

    batch_size = config.classification.batch_size
    rois = np.where(seg_map, rois, - IMAGENET_MEAN / IMAGENET_STD).astype(np.float32)
    labels = CLASSIFICATION_SESSION.batch_run(rois * seg_map, batch_size)
    labels = (labels.flatten() > 0).astype(np.uint8)

    # 응답 반환
    response = dict(
        bboxes=bboxes.tolist(),
        diseases=labels.tolist(),
        num_objects=len(bboxes)
    )

    return JSONResponse(content=response)
