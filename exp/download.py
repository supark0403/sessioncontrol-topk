"""Download the pinned model into exp/models (token from HF_TOKEN env, never stored)."""
import os
from huggingface_hub import snapshot_download

tok = os.environ.get("HF_TOKEN")
assert tok, "HF_TOKEN env var required"
snapshot_download(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    local_dir="models/Llama-3.1-8B-Instruct",
    token=tok,
    allow_patterns=["*.json", "*.safetensors", "tokenizer*", "special_tokens_map.json"],
    max_workers=8,
)
snapshot_download(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    local_dir="models/all-MiniLM-L6-v2",
    allow_patterns=["*.json", "*.safetensors", "*.txt", "1_Pooling/*"],
)
print("download ok")
