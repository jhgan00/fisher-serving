# 해수부 모델 파이프라인 & 서빙 

## Requirements

- 서버 관련 설정
  - 호스트, 포트, 파이썬 가상환경 경로 등을 설정
  - `ENV.sh` 을 사용하되, 해당 파일이 없으면 `ENV.sample.sh` 사용하도록 되어있음
- 모델 관련 설정
  - 모델 임계값, 디바이스, 배치 처리 사이즈 등을 설정
  - `config/gmission.proto.yml`

```bash
# python 3.8.10
pip install -r requirements.txt
bash run.gunicorn.sh

# 테스트 스크립트
python scripts/apitest.py
bash scripts/apitest.sh
```

# Endpoint

- `/detection`
  - 탐지된 광어 개체들의 바운딩 박스, 전체 몸통 여부, 질병 여부를 반환
  - 전체 몸통이 보이지 않은 개체들에 대해서는 질병 여부를 판단하지 않음