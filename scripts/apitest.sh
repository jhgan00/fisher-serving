# API 작동 테스트
while true; do

  curl -X 'POST' \
  'http://127.0.0.1:8000/detection' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@"/Users/jhgan/workspace/fisher-serving/resources/sample/20220816_100GOPRO_G0020073.JPG"' \
  -w "\n[*] Elapsed: %{time_total}\n"
done;