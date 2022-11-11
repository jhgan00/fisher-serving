# API 작동 테스트

if [[ -f ENV.sh ]]; then
  env_file=ENV.sh
else
  env_file=ENV.sample.sh
fi

echo "$env_file is sourced";
source $env_file
source $PYTHON_VENV/bin/activate

while true; do

  curl -X 'POST' \
  "http://$HOST:$PORT/detection" \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@"resources/sample/20220816_100GOPRO_G0020073.JPG"' \
  -w "\n[*] Elapsed: %{time_total}\n"

done;