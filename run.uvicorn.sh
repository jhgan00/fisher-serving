if [[ -f ENV.sh ]]; then
  env_file=ENV.sh
else
  env_file=ENV.sample.sh
fi
echo "$env_file is sourced";
source $env_file

uvicorn main:app --reload --host $HOST --port $PORT
