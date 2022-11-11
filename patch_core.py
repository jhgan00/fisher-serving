import os
import cv2
import torch
from torch import nn
import torchvision.transforms.functional as TF
from torch.nn import functional as F
from torchvision import transforms
import faiss

import numpy as np
from typing import Union
from scipy.ndimage import gaussian_filter


def clssification_preprocess(img):

    """
    RGB PIL.Image 를 입력받아 3 x 224 x 224 크기의 텐서를 출력
    """
    img = TF.to_tensor(img)
    img = TF.resize(img, (128, 128))
    img = TF.normalize(img, (0.485, 0.456, 0.406),
                      (0.229, 0.224, 0.225))
    return img


def anomaly_preprocess(img):
    """
    RGB PIL.Image 를 입력받아 3 x 224 x 224 크기의 텐서를 출력
    """
    img =TF.to_tensor(img)
    img =TF.resize(img, (224, 224))
    img =TF.normalize(img, (0.485, 0.456, 0.406),
                      (0.229, 0.224, 0.225))
    return img


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    #a_max = 11
    #a_min = 3
    return (image-a_min)/(a_max - a_min)


def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap


def heatmap_on_image(heatmap, image):
    if heatmap.shape != image.shape:
        heatmap = cv2.resize(heatmap, (image.shape[0], image.shape[1]))
    out = np.float32(heatmap[:,:,[2, 1, 0]])/255 + np.float32(image)/255
    out = out / np.max(out)
    return np.uint8(255 * out)


def save_anomaly_map(anomaly_map, input_img):
    if anomaly_map.shape != input_img.shape:
        anomaly_map = cv2.resize(anomaly_map, (input_img.shape[0], input_img.shape[1]))

    anomaly_map_norm = min_max_norm(anomaly_map)
    # anomaly map on image
    heatmap = cvt2heatmap(anomaly_map_norm * 255)
    hm_on_img = heatmap_on_image(heatmap, input_img)
    return input_img, heatmap, hm_on_img


class ResNetForPatchcoreEmbedding(nn.Module):

    """패치코어 임베딩 추출을 위한 ResNet 모델: 기존 코드의 self.model 에서 사용된 훅을 제거하고 다시 구현"""

    def __init__(self, repo_name: str, model_name: str):

        super().__init__()
        backbone = torch.hub.load(repo_name, model_name, pretrained=True)

        # first layer
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        # residual layers
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

        self.avg_pool = nn.AvgPool2d(3, 1, 1)

        for param in self.parameters():
            param.requires_grad = False

    def embedding_concat(self, x, y):
        """2nd, 3rd 레이어의 출력을 결합"""
        B, C1, H1, W1 = x.size()
        _, C2, H2, W2 = y.size()
        s = int(H1 / H2)
        x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
        x = x.view(B, C1, -1, H2, W2)
        z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
        for i in range(x.size(2)):
            z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
        z = z.view(B, -1, H2 * W2)
        z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)

        return z

    def forward(self, x: torch.Tensor):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        e1 = self.avg_pool(x)  # 2nd 레이어 피쳐

        x = self.layer3(x)
        e2 = self.avg_pool(x)  # 3rd 레이어 피쳐

        # 2nd 레이어와 3rd 레이어의 피쳐 결합
        z = self.embedding_concat(e1, e2)

        # reshape embedding 함수와 동일 기능: (batch_size, num_patches, embed_dim) 모양으로 재조정
        z = z.permute(0, 2, 3, 1).contiguous()
        z = torch.flatten(z, start_dim=1, end_dim=2)

        return z


class STPM(nn.Module):

    def __init__(self, repo_name:str, model_name: str, index_fpath: str, k: int, threshold: float, device_id: int):

        super(STPM, self).__init__()
        self.encoder = ResNetForPatchcoreEmbedding(repo_name, model_name)

        self.inv_normalize = transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
        )
        self.index = faiss.read_index(index_fpath)
        res = faiss.StandardGpuResources()
        self.index = faiss.index_cpu_to_gpu(res, device_id, self.index)

        self.k = k
        self.threshold = threshold
        self.embed_dim = self.index.d

    def forward(self, x: Union[torch.Tensor, np.ndarray]):
        """torch.Tensor 를 입력받아 anomlay score를 반환"""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)

        assert isinstance(x, torch.Tensor)

        # 피쳐 추출
        batch_size = x.size(0)
        x = self.encoder(x)

        # knn search: score_patches - 메모리 뱅크에서 가장 가까운 k개 정상 패치들과의 거리를 측정
        score_patches, _ = self.index.search(x.view(-1, self.embed_dim).detach().cpu().numpy(), k=self.k)

        # 배치 사이즈 x 패치 수 x K 크기로 리쉐입
        score_patches = score_patches.reshape(batch_size, -1, self.k)

        # 각 패치별로 메모리 뱅크에서 가장 가까운 피쳐와의 거리를 찾기
        closest_dist = score_patches[torch.arange(batch_size), :, 0]
        if batch_size == 1:  # 배치 사이즈가 1인 경우 차원이 스퀴즈됨: 이 경우 다시 차원 확장
            closest_dist = closest_dist[np.newaxis, :]

        # 가장 가까운 피쳐와의 거리가 가장 먼 패치를 찾기
        idxmax = closest_dist.argmax(axis=1)

        # 소프트맥스
        N_b = score_patches[torch.arange(batch_size), idxmax]
        N_b -= N_b.max(axis=1, keepdims=True)  # prevent overflow
        N_b_exp = np.exp(N_b)
        w = 1 - N_b_exp.max(axis=1) / N_b_exp.sum(axis=1)

        score = w * closest_dist.max(axis=1)

        # anomaly map 시각화 부분은 일단 제외함
        # anomaly_map = score_patches[torch.arange(batch_size), :, 0].reshape((batch_size, 28, 28))
        # # save images
        # inv_normalize = transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        #                                      std=[1 / 0.229, 1 / 0.224, 1 / 0.225])
        # i = 0
        # for _x, _anomaly_map in zip(x, anomaly_map):
        #     _x = inv_normalize(_x).unsqueeze(0)
        #     input_x = (_x.permute(0, 2, 3, 1).cpu().numpy()[0] * 255).astype(np.uint8)
        #
        #     _anomaly_map_resized = cv2.resize(_anomaly_map, (224, 224))
        #     _anomaly_map_resized_blur = gaussian_filter(_anomaly_map_resized, sigma=4)
        #     input_img, heatmap, hm_on_img = save_anomaly_map(_anomaly_map_resized_blur, input_x)
        #     cv2.imwrite(f"patchcore.{i}.jpg", cv2.cvtColor(hm_on_img, cv2.COLOR_RGB2BGR))
        #     i += 1

        return score

    def run(self, x):
        return self.__call__(x)

    def batch_run(self, x: Union[torch.Tensor, np.ndarray], batch_size: int):
        """
        - 입력을 배치 단위로 쪼개서 처리.
        - Onnxruntime BatchInference 세션과 인터페이스를 통일하기 위한 구현.
        """
        result = []
        for i in range(0, len(x), batch_size):
            result.append(self.run(x[i:i + batch_size]))
        result = np.concatenate(result)
        return result
