#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/vm_instance_config.sh"

echo "Configuring SSH..."
gcloud compute config-ssh --project "$PROJECT_ID"

echo "Creating remote directories..."
gcloud compute ssh --zone "$ZONE" "$INSTANCE" --project "$PROJECT_ID" -- \
  "mkdir -p ~/work/gqe-torch ~/tmp"

echo "Syncing project to VM..."
cd "$LOCAL_PROJECT_DIR"

rsync -avz --delete \
  --exclude='.venv' \
  --exclude='.venv-torch' \
  --exclude='.venv-jax' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.git' \
  ./ "$VM_USER@$INSTANCE.$ZONE.$PROJECT_ID:$REMOTE_DIR/"

echo "Setting up VM environment..."
gcloud compute ssh --zone "$ZONE" "$INSTANCE" --project "$PROJECT_ID" -- '
set -e

sudo apt update
sudo apt install -y python3 python3-pip python3-venv git rsync build-essential

cd ~/work/gqe-torch
mkdir -p ~/tmp
export TMPDIR=$HOME/tmp

rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel

if [ -f requirements.txt ]; then
  TMPDIR=$HOME/tmp pip install --no-cache-dir -r requirements.txt
fi

pip uninstall -y jax jaxlib jax-cuda13-plugin jax-cuda13-pjrt || true
pip uninstall -y "nvidia-*" || true

TMPDIR=$HOME/tmp pip install --no-cache-dir --upgrade "jax[cuda13]"

echo "Testing JAX GPU..."
python - <<PY
import jax
import jax.numpy as jnp

print("jax:", jax.__version__)
print("backend:", jax.default_backend())
print("devices:", jax.devices())

x = jnp.ones((1024, 1024))
y = x @ x
print("result device:", y.device)
PY

echo "Setup complete."
'

echo "Done."
echo "Run project with:"
echo "gcloud compute ssh --zone \"$ZONE\" \"$INSTANCE\" --project \"$PROJECT_ID\""
echo "cd ~/work/gqe-torch && source .venv/bin/activate && python main.py"
