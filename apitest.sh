if [[ -f ENV.sh ]]; then
  env_file=ENV.sh
else
  env_file=ENV.sample.sh
fi

echo "$env_file is sourced";
source $env_file

# API 작동 테스트
head -n 10 input.sample.txt | while read line; do
  echo $line
  curl -X 'POST' \
  'http://127.0.0.1:12000/disease' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"img_fpath":"'$line'"}' \
  -o /dev/null \
  -w "Elapsed: %{time_starttransfer}\n" \
  -s

done
