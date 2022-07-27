import sys

import os

import yaml
from easydict import EasyDict
from fastapi import FastAPI
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse

from detection.efficientdet.efficientnet import EfficientNet
from detection.efficientdet import EfficientDetBackbone
from detection.efficientdet.utils import BBoxTransform, ClipBoxes
from detection.utils import *
from pydantic import BaseModel


class Item(BaseModel):
    img_fpath: str


# 설정파일 로드
params = yaml.safe_load(open(os.environ.get("MODEL_CONFIG_FPATH")).read())
params = EasyDict(params)

# get detection model
detector = EfficientDetBackbone(
    num_classes=len(params.detection.obj_list),
    compound_coef=params.detection.compound_coef,
    ratios=eval(params.detection.anchors_ratios),
    scales=eval(params.detection.anchors_scales)
)
state_dict = torch.load(params.detection.state_dict_path)
detector.load_state_dict(state_dict)
detector.eval()
detector.requires_grad_(False)
device = torch.device(os.environ.get("DEVICE"))
detector.to(device)

# get classification model
classifier = EfficientNet.from_pretrained(f'efficientnet-b{params.classification.compound_coef}', False, num_classes=2)
state_dict = torch.load(params.classification.state_dict_path)
classifier.load_state_dict(state_dict)
classifier.eval()
classifier.requires_grad_(False)
device = torch.device(os.environ.get("DEVICE"))
classifier.to(device)

app = FastAPI()


@torch.no_grad()
@app.post("/disease")
async def get_bboxes_and_diseases(item: Item) -> JSONResponse:
    """
    요청된 이미지에 대해 넙치 객체의 바운딩 박스, 키포인트, 질병 여부를 반환
    - 바운딩 박스: x1, y1, x2, y2 순서 (Top left & bottom right)
    - 키포인트: 입 x, 입 y, 꼬리x, 꼬리 y, 등 x, 등 y, 배 x, 배 y
    # TODO. 현재는 fisher 서버의 validation 이미지 경로를 읽도록 되어있음. REST API 에서 어떤 식으로 이미지 교환하는지 알아보고 수정 필요

    :param img_fpath: 처리할 이미지 경로 (fisher 서버)
    :return: JSONResponse

    """

    # 요청된 프레임 읽기
    img_fpath = os.path.join(os.environ.get("IMAGE_DIR"), item.img_fpath)

    sys.stdout.write(img_fpath + "\n")

    # 이미지 읽기: 자원이 없으면 404 반환
    frame = cv2.imread(img_fpath)
    if frame is None:
        sys.stderr.write(img_fpath + "\n")
        raise HTTPException(status_code=404, detail=f"File not found: {img_fpath}")

    ####################################################################################################################
    # 1. Detection
    ####################################################################################################################
    # 디텍션 모델을 위한 전처리 수행
    ori_imgs, framed_imgs, framed_metas = preprocess_video(frame, max_size=params.detection.input_size, mean=params.mean,
                                                           std=params.std)
    x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)
    x = x.to(device).permute(0, 3, 1, 2)

    # 예측 계산: 바운딩 박스, 클래스 확률, 키포인트
    _, regression, classification, anchors, regression_kp = detector(x)

    # 디텍션 결과에 대한 후처리
    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()
    out = postprocess_with_KP(x, anchors, regression, regression_kp, classification, regressBoxes, clipBoxes,
                              params.detection.threshold, params.detection.iou_threshold)
    out = invert_affine(framed_metas, out)
    out = filtering_overlap(out, params.detection.score_threshold)[0]

    out['rois'] = out['rois'].astype(np.int)  # ROI: x1, y1, x2, y2 순서

    # Keypoint: Pixel space 에서의 좌표로 변환
    for i in range(len(out)):

        kp, roi = out['kps'][i], out['rois'][i]
        x1, y1, x2, y2 = roi

        out['kps'][i][:] = np.array([
            # KP 1
            int(kp[0] * (x2 - x1 + 1) + x1 + 0.5),
            int(kp[1] * (y2 - y1 + 1) + y1 + 0.5),

            # KP 2
            int(kp[2] * (x2 - x1 + 1) + x1 + 0.5),
            int(kp[3] * (y2 - y1 + 1) + y1 + 0.5),

            # KP 3
            int(kp[4] * (x2 - x1 + 1) + x1 + 0.5),
            int(kp[5] * (y2 - y1 + 1) + y1 + 0.5),

            # KP 3
            int(kp[6] * (x2 - x1 + 1) + x1 + 0.5),
            int(kp[7] * (y2 - y1 + 1) + y1 + 0.5)
        ])
    out['kps'] = out['kps'].astype(int)

    ####################################################################################################################

    ####################################################################################################################
    # 2. Classification
    ####################################################################################################################

    # 전처리
    # 디텍션 모델의 결과를 활용하여 바운딩 박스 크롭하고 배치로 묶기
    normalized_img = (frame / 255 - params.mean) / params.std
    cropped_imgs = []
    for (x1, y1, x2, y2) in out['rois']:
        input_size = params.classification.input_size
        # TODO: aspect_aware padding 이후 리사이징하도록 수정
        cropped_img = cv2.resize(normalized_img[y1:y2+1, x1:x2+1], (input_size, input_size))
        cropped_imgs.append(cropped_img)
    x = torch.stack([torch.from_numpy(img) for img in cropped_imgs], 0)
    x = x.to(device).permute(0, 3, 1, 2).type(torch.float)

    # 추론
    outputs = classifier(x)
    disease = outputs.argmax(-1).cpu().numpy()
    ####################################################################################################################

    # 응답 반환
    response = dict(
        bboxes=out['rois'].tolist(),
        keypoints=out['kps'].tolist(),
        class_ids=out['class_ids'].tolist(),
        scores=out['scores'].tolist(),
        diseases=disease.tolist()
    )

    return JSONResponse(content=response)
