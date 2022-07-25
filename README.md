# 해수부 모델 파이프라인 & 서빙 

## Notice

- API 작동은 테스트해봤지만 **예측이 제대로 작동하고 있는지는 아직 테스트 안해봤음**
  - 다양한 오류 및 실수가 있을 수 있음
  - 실제로 결과 받아서 그려봐야 할듯
- 경로 설정이 fisher 서버 기준으로 되어있음
  - `/home/fisher/Peoples/jhgan/fisher-serving`

## Requirements

```bash
# fisher 서버
# python 3.9.12
pip install -r requirements.txt
bash run.uvicorn.sh
bash apitest.sh  # 서버 시작된 이후 API 작동 테스트
```

