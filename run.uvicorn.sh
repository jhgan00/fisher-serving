if [[ -f ENV.sh ]]; then
  env_file=ENV.sh
else
  env_file=ENV.sample.sh
fi

echo "$env_file is sourced";
source $env_file
source $PYTHON_VENV/bin/activate

if [ $DEBUG = false ]; then
  uvicorn main:app --host $HOST --port $PORT --workers $WEB_CONCURRENCY --log-level info
else
  uvicorn main:app --reload --host $HOST --port $PORT --log-level debug
fi;
