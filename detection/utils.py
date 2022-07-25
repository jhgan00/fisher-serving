#Author: Zylo117

from typing import Union

import cv2
import numpy as np
import torch
from torchvision.ops.boxes import batched_nms
from copy import deepcopy


def invert_affine(metas: Union[float, list, tuple], preds):
    for i in range(len(preds)):
        if len(preds[i]['rois']) == 0:
            continue
        else:
            if metas is float:
                preds[i]['rois'][:, [0, 2]] = preds[i]['rois'][:, [0, 2]] / metas
                preds[i]['rois'][:, [1, 3]] = preds[i]['rois'][:, [1, 3]] / metas
            else:
                new_w, new_h, old_w, old_h, padding_w, padding_h = metas[i]
                preds[i]['rois'][:, [0, 2]] = preds[i]['rois'][:, [0, 2]] / (new_w / old_w)
                preds[i]['rois'][:, [1, 3]] = preds[i]['rois'][:, [1, 3]] / (new_h / old_h)
    return preds


def aspectaware_resize_padding(image, width, height, interpolation=None, means=None):
    old_h, old_w, c = image.shape
    if old_w > old_h:
        new_w = width
        new_h = int(width / old_w * old_h)
    else:
        new_w = int(height / old_h * old_w)
        new_h = height

    canvas = np.zeros((height, height, c), np.float32)
    if means is not None:
        canvas[...] = means

    if new_w != old_w or new_h != old_h:
        if interpolation is None:
            image = cv2.resize(image, (new_w, new_h))
        else:
            image = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    padding_h = height - new_h
    padding_w = width - new_w

    if c > 1:
        canvas[:new_h, :new_w] = image
    else:
        if len(image.shape) == 2:
            canvas[:new_h, :new_w, 0] = image
        else:
            canvas[:new_h, :new_w] = image

    return canvas, new_w, new_h, old_w, old_h, padding_w, padding_h,


def preprocess(*image_path, max_size=512, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    ori_imgs = [cv2.imread(img_path)[..., ::-1] for img_path in image_path]
    normalized_imgs = [(img / 255 - mean) / std for img in ori_imgs]
    imgs_meta = [aspectaware_resize_padding(img, max_size, max_size,
                                            means=None) for img in normalized_imgs]
    framed_imgs = [img_meta[0] for img_meta in imgs_meta]
    framed_metas = [img_meta[1:] for img_meta in imgs_meta]

    return ori_imgs, framed_imgs, framed_metas


def preprocess_video(*frame_from_video, max_size=512, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)):
    ori_imgs = frame_from_video
    normalized_imgs = [(img / 255 - mean) / std for img in ori_imgs]
    imgs_meta = [aspectaware_resize_padding(img[..., ::-1], max_size, max_size,
                                            means=None) for img in normalized_imgs]
    framed_imgs = [img_meta[0] for img_meta in imgs_meta]
    framed_metas = [img_meta[1:] for img_meta in imgs_meta]

    return ori_imgs, framed_imgs, framed_metas


def postprocess(x, anchors, regression, classification, regressBoxes, clipBoxes, threshold, iou_threshold):
    transformed_anchors = regressBoxes(anchors, regression)
    transformed_anchors = clipBoxes(transformed_anchors, x)
    scores = torch.max(classification, dim=2, keepdim=True)[0]
    scores_over_thresh = (scores > threshold)[:, :, 0]
    out = []
    for i in range(x.shape[0]):
        if scores_over_thresh[i].sum() == 0:
            out.append({
                'rois': np.array(()),
                'class_ids': np.array(()),
                'scores': np.array(()),
            })
            continue

        classification_per = classification[i, scores_over_thresh[i, :], ...].permute(1, 0)
        transformed_anchors_per = transformed_anchors[i, scores_over_thresh[i, :], ...]
        scores_per = scores[i, scores_over_thresh[i, :], ...]
        scores_, classes_ = classification_per.max(dim=0)
        anchors_nms_idx = batched_nms(transformed_anchors_per, scores_per[:, 0], classes_, iou_threshold=iou_threshold)

        if anchors_nms_idx.shape[0] != 0:
            classes_ = classes_[anchors_nms_idx]
            scores_ = scores_[anchors_nms_idx]
            boxes_ = transformed_anchors_per[anchors_nms_idx, :]

            out.append({
                'rois': boxes_.cpu().numpy(),
                'class_ids': classes_.cpu().numpy(),
                'scores': scores_.cpu().numpy(),
            })
        else:
            out.append({
                'rois': np.array(()),
                'class_ids': np.array(()),
                'scores': np.array(()),
            })

    return out


def postprocess_with_KP(x, anchors, regression, regression_kp, classification, regressBoxes, clipBoxes, threshold,
                        iou_threshold):
    transformed_anchors = regressBoxes(anchors, regression)
    transformed_anchors = clipBoxes(transformed_anchors, x)
    scores = torch.max(classification, dim=2, keepdim=True)[0]
    scores_over_thresh = (scores > threshold)[:, :, 0]
    out = []
    for i in range(x.shape[0]):
        if scores_over_thresh[i].sum() == 0:
            out.append({
                'rois': np.array(()),
                'class_ids': np.array(()),
                'scores': np.array(()),
            })
            continue

        classification_per = classification[i, scores_over_thresh[i, :], ...].permute(1, 0)
        transformed_anchors_per = transformed_anchors[i, scores_over_thresh[i, :], ...]
        regression_kp_per = regression_kp[i, scores_over_thresh[i, :], ...]
        scores_per = scores[i, scores_over_thresh[i, :], ...]
        scores_, classes_ = classification_per.max(dim=0)
        anchors_nms_idx = batched_nms(transformed_anchors_per, scores_per[:, 0], classes_, iou_threshold=iou_threshold)

        if anchors_nms_idx.shape[0] != 0:
            classes_ = classes_[anchors_nms_idx]
            scores_ = scores_[anchors_nms_idx]
            boxes_ = transformed_anchors_per[anchors_nms_idx, :]
            keypoints_ = regression_kp_per[anchors_nms_idx, :]
            out.append({
                'rois': boxes_.cpu().numpy(),
                'class_ids': classes_.cpu().numpy(),
                'scores': scores_.cpu().numpy(),
                'kps': keypoints_.cpu().numpy()
            })
        else:
            out.append({
                'rois': np.array(()),
                'class_ids': np.array(()),
                'scores': np.array(()),
                'kps': keypoints_.cpu().numpy()
            })

    return out


def postprocess_on_training(x, anchors, regression, classification, regressBoxes, clipBoxes, threshold, iou_threshold):
    transformed_anchors = regressBoxes(anchors, regression)
    transformed_anchors = clipBoxes(transformed_anchors, x)
    scores = torch.max(classification, dim=2, keepdim=True)[0]
    scores_over_thresh = (scores > threshold)[:, :, 0]
    out = []
    for i in range(x.shape[0]):
        if scores_over_thresh[i].sum() == 0:
            out.append({
                'rois': np.array(()),
                'class_ids': np.array(()),
                'scores': np.array(()),
            })
            continue

        classification_per = classification[i, scores_over_thresh[i, :], ...].permute(1, 0)
        transformed_anchors_per = transformed_anchors[i, scores_over_thresh[i, :], ...]
        scores_per = scores[i, scores_over_thresh[i, :], ...]
        scores_, classes_ = classification_per.max(dim=0)
        anchors_nms_idx = batched_nms(transformed_anchors_per, scores_per[:, 0], classes_, iou_threshold=iou_threshold)

        if anchors_nms_idx.shape[0] != 0:
            classes_ = classes_[anchors_nms_idx]
            scores_ = scores_[anchors_nms_idx]
            boxes_ = transformed_anchors_per[anchors_nms_idx, :]

            out.append({
                'rois': boxes_.cpu().detach().numpy(),
                'class_ids': classes_.cpu().detach().numpy(),
                'scores': scores_.cpu().detach().numpy(),
            })
        else:
            out.append({
                'rois': np.array(()),
                'class_ids': np.array(()),
                'scores': np.array(()),
            })

    return out


def filtering_overlap(out, score_thres):
    deletex_bbx_idx = []
    iouMap_detect = []
    for i in range(out[0]['rois'].shape[0]):
        iouMax = 0
        idx = -1
        for j in range(out[0]['rois'].shape[0]):
            if i == j:
                continue
            x1 = max(out[0]['rois'][i][0], out[0]['rois'][j][0])
            y1 = max(out[0]['rois'][i][1], out[0]['rois'][j][1])
            x2 = min(out[0]['rois'][i][2], out[0]['rois'][j][2])
            y2 = min(out[0]['rois'][i][3], out[0]['rois'][j][3])

            if x2 - x1 > 0 and y2 - y1:
                overlapArea = (x2 - x1) * (y2 - y1)
                areaTracking = (out[0]['rois'][i][2] - out[0]['rois'][i][0]) * (
                        out[0]['rois'][i][3] - out[0]['rois'][i][1])
                areaDetection = (out[0]['rois'][j][2] - out[0]['rois'][j][0]) * (
                        out[0]['rois'][j][3] - out[0]['rois'][j][1])
                iou = (overlapArea) / (areaTracking + areaDetection - overlapArea)
                if iou > iouMax:
                    iouMax = iou
                    idx = j

        iouMap_detect.append([i, idx, iouMax])

    out_filtered = deepcopy(out)
    idx_0 = -1
    idx_1 = -1
    iouMapIdx = 0
    cnt = 0

    while iouMapIdx < len(iouMap_detect):
        if iouMap_detect[iouMapIdx][2] > 0.85:
            idx_0 = iouMap_detect[iouMapIdx][0]
            idx_1 = iouMap_detect[iouMapIdx][1]
            iouMap_detect.pop(iouMapIdx)

            score_idx_0 = out_filtered[0]['scores'][idx_0]
            score_idx_1 = out_filtered[0]['scores'][idx_1]

            if score_idx_0 > score_idx_1:
                deletex_bbx_idx.append(idx_1)

                idx_del = idx_1
                idx_keep = idx_0
            else:
                deletex_bbx_idx.append(idx_0)

                idx_del = idx_0
                idx_keep = idx_1

            deletex_bbx_idx.append(idx_del)
            iouMapIdx = iouMapIdx - 1

            iouMapIdx2 = 0
            target_idx_to_del = -1
            while iouMapIdx2 < len(iouMap_detect):
                if (len(iouMap_detect) > iouMapIdx2 and len(iouMap_detect) > 0):  # list에 아무것도 없을 때의 케이스 고려
                    if (iouMap_detect[iouMapIdx2][0] == idx_del and iouMap_detect[iouMapIdx2][1] == idx_keep):
                        iouMap_detect.pop(iouMapIdx2)
                        iouMapIdx2 = iouMapIdx2 - 1

                if (len(iouMap_detect) > iouMapIdx2 and len(iouMap_detect) > 0):  # list에 아무것도 없을 때의 케이스 고려
                    if (iouMap_detect[iouMapIdx2][0] == idx_del):
                        if iouMap_detect[iouMapIdx2][2] > score_thres:
                            deletex_bbx_idx.append(iouMap_detect[iouMapIdx2][1])
                            iouMap_detect.pop(iouMapIdx2)
                            iouMapIdx2 = iouMapIdx2 - 1

                if (len(iouMap_detect) > iouMapIdx2 and len(iouMap_detect) > 0):  # list에 아무것도 없을 때의 케이스 고려
                    if (iouMap_detect[iouMapIdx2][1] == idx_del):
                        if iouMap_detect[iouMapIdx2][2] > score_thres:
                            deletex_bbx_idx.append(iouMap_detect[iouMapIdx2][0])
                            iouMap_detect.pop(iouMapIdx2)
                            iouMapIdx2 = iouMapIdx2 - 1

                iouMapIdx2 = iouMapIdx2 + 1

        iouMapIdx = iouMapIdx + 1

    for i in range(len(deletex_bbx_idx)):
        target_idx_to_del = deletex_bbx_idx[i]
        out_filtered_idx = 0
        while out_filtered_idx < out_filtered[0]['scores'].shape[0]:
            if out_filtered[0]['rois'][out_filtered_idx][0] == out[0]['rois'][target_idx_to_del][0] and \
                    out_filtered[0]['rois'][out_filtered_idx][1] == out[0]['rois'][target_idx_to_del][1] and \
                    out_filtered[0]['rois'][out_filtered_idx][2] == out[0]['rois'][target_idx_to_del][2] and \
                    out_filtered[0]['rois'][out_filtered_idx][3] == out[0]['rois'][target_idx_to_del][3]:
                out_filtered[0]['scores'] = np.delete(out_filtered[0]['scores'], (out_filtered_idx), axis=0)
                out_filtered[0]['rois'] = np.delete(out_filtered[0]['rois'], (out_filtered_idx), axis=0)
                out_filtered[0]['kps'] = np.delete(out_filtered[0]['kps'], (out_filtered_idx), axis=0)
                out_filtered[0]['class_ids'] = np.delete(out_filtered[0]['class_ids'], (out_filtered_idx), axis=0)
                out_filtered_idx = out_filtered_idx - 1
            out_filtered_idx = out_filtered_idx + 1

    out = deepcopy(out_filtered)
    return out
