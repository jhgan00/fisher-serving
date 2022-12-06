import io
import os
from typing import List

import cv2
import numpy as np
import torch
import yaml
from easydict import EasyDict
from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse

from utils import non_max_suppression, scale_coords, filter_invalid_bboxes, letterbox
from utils import IMAGENET_MEAN, IMAGENET_STD
from utils import sigmoid, encode_base64, postprocess_anomaly_maps
from inference_session import initialize_session
from patch_core import STPM


# 설정파일 로드
MODEL_CONFIG_FPATH = os.environ.get("MODEL_CONFIG_FPATH")
config = yaml.safe_load(open(MODEL_CONFIG_FPATH).read())
config = EasyDict(config)

# 모델 초기화
DETECTION_SESSION = initialize_session(config.detection.model_path, config.detection.device_id)
FISH_CLASSIFICATION_SESSION = initialize_session(config.fish_classification.model_path, config.fish_classification.device_id)
SEGMENTATION_SESSION = initialize_session(config.segmentation.model_path, config.segmentation.device_id)
CLASSIFICATION_SESSION = initialize_session(config.classification.model_path, config.classification.device_id)
ANOMALY_SESSION = initialize_session(config.anomaly.model_path, config.anomaly.device_id)
PATCHCORE_SESSION = STPM(
    config.patchcore.repo_name,
    config.patchcore.model_name,
    config.patchcore.index_path,
    config.patchcore.k,
    config.patchcore.threshold,
    config.patchcore.device_id,
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
    img = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    img = letterbox(img, (config.detection.input_size, config.detection.input_size), auto=False)
    img = img.astype(np.float32).transpose(2, 0, 1)[np.newaxis, ...]
    img /= 255.

    output = DETECTION_SESSION.run(None, {'images': img})[0]
    output = torch.tensor(output)
    output = non_max_suppression(output, config.detection.conf_thres, config.detection.iou_thres)[0]
    conf_scores = output[:, 4]
    bboxes = scale_coords(img.shape[2:], output[:, :4], org_shape).round().type(torch.int)
    bboxes, conf_scores = filter_invalid_bboxes(bboxes, conf_scores)

    num_objects = len(bboxes)

    if not num_objects:
        content = dict(
            bboxes=[],
            conf_scores=[],
            diseases=[],
            anomaly_scores=[],
            anomaly_maps=[],
            whole_shape=[],
            fish_category=[],
            num_objects=num_objects
        )
        return JSONResponse(content=content)

    anomaly_scores = np.array([0. for _ in range(num_objects)], dtype=np.float32)
    anomaly_maps = np.array(["" for _ in range(num_objects)], dtype=object)
    diseases = np.array([0 for _ in range(num_objects)], dtype=np.uint8)
    is_whole_shape = np.array([0 for _ in range(num_objects)], dtype=np.uint8)
    mask = np.array([False for _ in range(num_objects)], dtype=np.bool)

    # extract rois from the original input while preserving aspects
    rois: List[np.ndarray] = []
    img = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)  # RGB order
    for i, (xmin, ymin, xmax, ymax) in enumerate(bboxes):
        roi = img[ymin:ymax + 1, xmin:xmax + 1, :]
        rois.append(roi)

    ####################################################################################################################
    # 2. Fish category classification
    ####################################################################################################################

    # classification 모델 입력에 맞게 resize
    dsize = (config.fish_classification.input_size, config.fish_classification.input_size)
    classification_rois = np.stack([letterbox(roi, new_shape=dsize, color=(0, 0, 0)) for roi in rois]).astype(np.float32)
    classification_rois = classification_rois.transpose(0, 3, 1, 2)  # B x C x H x W
    classification_rois /= 255.
    classification_rois -= IMAGENET_MEAN
    classification_rois /= IMAGENET_STD
    fish_category = FISH_CLASSIFICATION_SESSION.batch_run(classification_rois, config.fish_classification.batch_size).argmax(axis=1)
    fish_category_mask = (fish_category == config.fish_classification.target_index)
    mask[:] = fish_category_mask

    if not fish_category_mask.any():
        content = dict(
            bboxes=bboxes.tolist(),
            conf_scores=conf_scores.tolist(),
            diseases=diseases.tolist(),
            is_whole_shape=is_whole_shape.tolist(),
            anomaly_scores=anomaly_scores.tolist(),
            anomaly_maps=anomaly_maps.tolist(),
            num_objects=num_objects,
            fish_category=fish_category.tolist()
        )
        return JSONResponse(content=content)

    ####################################################################################################################
    # 2. Segmentation
    ####################################################################################################################

    # run segmentation session
    dsize = (config.segmentation.input_size, config.segmentation.input_size)
    segmentation_rois = np.stack([letterbox(roi, new_shape=dsize, color=(0, 0, 0)) for roi in rois]).astype(np.float32)
    segmentation_rois = segmentation_rois.transpose(0, 3, 1, 2)  # B x C x H x W
    segmentation_rois = segmentation_rois[mask]
    segmentation_rois /= 255.
    segmentation_map = SEGMENTATION_SESSION.batch_run(segmentation_rois, config.segmentation.batch_size)
    segmentation_map = segmentation_map > config.segmentation.threshold  # B x C x H x W

    ####################################################################################################################
    # 3. Classification: 광어 몸통 전체인지를 분류
    ####################################################################################################################

    classification_rois = segmentation_rois * segmentation_map  # apply segmentation mask (B x C x H x W)
    classification_rois = classification_rois.transpose(0, 2, 3, 1)  # (B x H x W x C)
    dsize = (config.classification.input_size, config.classification.input_size)
    classification_rois = np.stack([cv2.resize(roi, dsize=dsize) for roi in classification_rois])
    classification_rois = classification_rois.transpose(0, 3, 1, 2)  # B x C x H x W
    classification_rois -= IMAGENET_MEAN
    classification_rois /= IMAGENET_STD
    logits = CLASSIFICATION_SESSION.batch_run(classification_rois, config.classification.batch_size).flatten()
    logits /= config.classification.temperature
    probs = sigmoid(logits)
    whole_shape_mask = (probs > config.classification.threshold)  # sigmoid & threshold 적용하도록 수정
    mask[fish_category_mask] = whole_shape_mask
    is_whole_shape[mask] = 1

    ####################################################################################################################
    # 4. Anomaly detection
    ####################################################################################################################

    if whole_shape_mask.any():

        anomaly_rois = segmentation_rois[whole_shape_mask] * segmentation_map[whole_shape_mask]
        dsize = (config.anomaly.input_size, config.anomaly.input_size)
        anomaly_rois = anomaly_rois.transpose(0, 2, 3, 1)  # (B x H x W x C)
        anomaly_rois = np.stack([cv2.resize(roi, dsize=dsize) for roi in anomaly_rois])
        anomaly_rois = anomaly_rois.transpose(0, 3, 1, 2)  # (B x C x H x W)
        anomaly_rois -= IMAGENET_MEAN
        anomaly_rois /= IMAGENET_STD

        scores = ANOMALY_SESSION.batch_run(anomaly_rois, config.anomaly.batch_size)
        scores = sigmoid(scores.squeeze())

        _, heatmaps = PATCHCORE_SESSION.batch_run(anomaly_rois, config.anomaly.batch_size)
        heatmaps = postprocess_anomaly_maps(heatmaps, dsize=(config.segmentation.input_size, config.segmentation.input_size))
        heatmaps *= segmentation_map[whole_shape_mask]
        anomaly_maps[mask] = [encode_base64(heatmap) for heatmap in heatmaps.transpose(0, 2, 3, 1)]
        anomaly_scores[mask] = scores
        diseases[mask] = (scores > config.anomaly.threshold).astype(np.uint8)
        is_whole_shape[mask] = 1

    # 응답 반환
    content = dict(
        bboxes=bboxes.tolist(),
        conf_scores=conf_scores.tolist(),
        diseases=diseases.tolist(),
        is_whole_shape=is_whole_shape.astype(np.uint8).tolist(),
        anomaly_scores=anomaly_scores.tolist(),
        anomaly_maps=anomaly_maps.tolist(),
        num_objects=len(bboxes),
        fish_category=fish_category.tolist()
    )

    return JSONResponse(content=content)
