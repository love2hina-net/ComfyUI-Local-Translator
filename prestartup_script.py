import sys
from pathlib import Path

from huggingface_hub import snapshot_download

path = (Path(__file__).parent).resolve()
sys.path.append(str(path))

import common

def download_model():
    if common.MODEL_PATH.exists():
        print(f"[Local Translator] Local Language Model was exited. installing LLM is not required. model: {common.MODEL_PATH}")
    else:
        print('[Local Translator] Downloading Local Language Model...')
        snapshot_download(repo_id=common.MODEL_REPOID, local_dir=common.MODEL_PATH)

print('[Local Translator] Prestartup started...')
download_model()
print('[Local Translator] Prestartup completed.')
