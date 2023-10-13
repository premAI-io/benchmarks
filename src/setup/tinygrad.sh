# source this!
git clone --depth=1 https://github.com/tinygrad/tinygrad.git /tmp/tinygrad

# setup env
python -m venv /tmp/tinygrad/.venv
source /tmp/tinygrad/.venv/bin/activate
python3 -m pip install -e /tmp/tinygrad
