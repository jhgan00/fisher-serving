import io
import os

import cv2
import numpy as np
import torch
import yaml
from easydict import EasyDict
from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse

from utils import non_max_suppression, scale_coords, filter_invalid_bboxes, letterbox, IMAGENET_MEAN, IMAGENET_STD, sigmoid
from inference_session import initialize_session
from patch_core import STPM

# 설정파일 로드
MODEL_CONFIG_FPATH = os.environ.get("MODEL_CONFIG_FPATH")
config = yaml.safe_load(open(MODEL_CONFIG_FPATH).read())
config = EasyDict(config)

# 모델 초기화
DETECTION_SESSION = initialize_session(config.detection.model_path, config.detection.device_id)
SEGMENTATION_SESSION = initialize_session(config.segmentation.model_path, config.segmentation.device_id)
CLASSIFICATION_SESSION = initialize_session(config.classification.model_path, config.classification.device_id)
ANOMALY_SESSION = STPM(
    config.anomaly.repo_name,
    config.anomaly.model_name,
    config.anomaly.index_path,
    config.anomaly.k,
    config.anomaly.threshold,
    config.anomaly.device_id,
)

app = FastAPI()


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

    # 모델 인풋 사이즈에 맞춰서 리사이징 (가로:세로 비율 보존)
    img = letterbox(img0, (config.detection.input_size, config.detection.input_size), auto=False)[0]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32).transpose(2, 0, 1)[np.newaxis, ...]
    img /= 255.

    output = DETECTION_SESSION.run(None, {'images': img})[0]
    output = torch.tensor(output)
    output = non_max_suppression(output, config.detection.conf_thres, config.detection.iou_thres)[0]
    conf_scores = output[:, 4]
    bboxes = scale_coords(img.shape[2:], output[:, :4], org_shape).round().type(torch.int)
    bboxes, conf_scores = filter_invalid_bboxes(bboxes, conf_scores)

    if not len(bboxes):
        response = dict(
            bboxes=[],
            conf_scores=[],
            diseases=[],
            num_objects=0
        )
        return JSONResponse(content=response)

    ####################################################################################################################
    # 2. Segmentation
    ####################################################################################################################

    # extract ROIs: preserve aspects
    rois = []
    img = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    for i, (xmin, ymin, xmax, ymax) in enumerate(bboxes):
        roi = img[ymin:ymax + 1, xmin:xmax + 1, :]
        roi, _, _ = letterbox(roi, new_shape=(config.segmentation.input_size, config.segmentation.input_size), color=(0, 0, 0))
        rois.append(roi)
    rois = np.stack(rois).astype(np.float32)
    rois = rois.transpose(0, 3, 1, 2)  # B x C x H x W
    rois /= 255.

    # run segmentation session
    seg_map = SEGMENTATION_SESSION.batch_run(rois, config.segmentation.batch_size)
    seg_map = seg_map > config.segmentation.threshold  # B x C x H x W

    ####################################################################################################################
    # 3. Classification: 광어 몸통 전체인지를 분류
    ####################################################################################################################

    rois = rois * seg_map  # apply segmentation mask (B x C x H x W)
    rois = rois.transpose(0, 2, 3, 1)  # (B x H x W x C)

    # classification 모델 입력에 맞게 resize
    classification_rois = np.stack([cv2.resize(roi, dsize=(config.classification.input_size, config.classification.input_size)) for roi in rois])
    classification_rois = classification_rois.transpose(0, 3, 1, 2)  # B x C x H x W
    classification_rois -= IMAGENET_MEAN
    classification_rois /= IMAGENET_STD
    logits = CLASSIFICATION_SESSION.batch_run(classification_rois, config.classification.batch_size).flatten()
    probs = sigmoid(logits)
    mask = probs > config.classification.threshold  # sigmoid & threshold 적용하도록 수정

    ####################################################################################################################
    # 4. Anomaly detection
    ####################################################################################################################

    anomaly_scores = np.array([0. for _ in range(len(rois))], dtype=np.float32)
    labels = np.array([0 for _ in range(len(rois))], dtype=np.uint8)
    if mask.any():
        anomaly_rois = rois[mask]  # 광어의 몸통 전체가 보이는 ROI 만을 선택 (B x H x W x C)
        anomaly_rois = np.stack([cv2.resize(roi, dsize=(config.anomaly.input_size, config.anomaly.input_size)) for roi in anomaly_rois])
        anomaly_rois = anomaly_rois.transpose(0, 3, 1, 2)  # (B x C x H x W)
        anomaly_rois -= IMAGENET_MEAN
        anomaly_rois /= IMAGENET_STD
        _anomaly_scores = ANOMALY_SESSION.batch_run(anomaly_rois, config.anomaly.batch_size)
        labels[mask] = (_anomaly_scores > config.anomaly.threshold).astype(np.uint8)
        anomaly_scores[mask] = _anomaly_scores

    # 응답 반환
    response = dict(
        bboxes=bboxes.tolist(),
        conf_scores=conf_scores.tolist(),
        diseases=labels.tolist(),
        is_whole_body=mask.astype(np.uint8).tolist(),
        anomaly_scores=anomaly_scores.tolist(),
        num_objects=len(bboxes)
    )

    return JSONResponse(content=response)
