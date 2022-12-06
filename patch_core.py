import cv2
import torch
from torch import nn
import torchvision.transforms.functional as TF
from torch.nn import functional as F
from torchvision import transforms
import faiss

import numpy as np
from typing import Union


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

    @torch.no_grad()
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
        # res = faiss.StandardGpuResources()
        # self.index = faiss.index_cpu_to_gpu(res, device_id, self.index)

        self.k = k
        self.threshold = threshold
        self.embed_dim = self.index.d

    @torch.no_grad()
    def forward(self, x: Union[torch.Tensor, np.ndarray]):

        """torch.Tensor 를 입력받아 anomlay score를 반환"""

        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x_org = x

        assert isinstance(x, torch.Tensor)

        # 피쳐 추출
        batch_size = x.size(0)

        with torch.cuda.amp.autocast():
            x = self.encoder(x)
        x = x.type(torch.float32)

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
        anomaly_maps = score_patches[torch.arange(batch_size), :, 0].reshape((batch_size, 28, 28))

        return score, anomaly_maps  # visualizations

    def run(self, x):
        return self.__call__(x)

    def batch_run(self, x: Union[torch.Tensor, np.ndarray], batch_size: int):
        """
        - 입력을 배치 단위로 쪼개서 처리.
        - Onnxruntime BatchInference 세션과 인터페이스를 통일하기 위한 구현.
        """
        scores = []
        visualizations = []
        for i in range(0, len(x), batch_size):
            score, vis = self.run(x[i:i + batch_size])
            scores.append(score)
            visualizations.append(vis)
        scores = np.concatenate(scores)
        visualizations = np.concatenate(visualizations)
        return scores, visualizations
