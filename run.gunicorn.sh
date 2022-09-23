if [[ -f ENV.sh ]]; then
  env_file=ENV.sh
else
  env_file=ENV.sample.sh
fi

echo "$env_file is sourced";
source $env_file
source $PYTHON_VENV/bin/activate

gunicorn main:app \
-k uvicorn.workers.UvicornWorker \
--access-logfile ./gunicorn-access.log \
--bind $HOST:$PORT \
--workers $WEB_CONCURRENCY \
--daemon
