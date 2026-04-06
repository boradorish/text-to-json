from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="boradorish/qwen3-0.6b-finetuned",
    local_dir="model/qwen3-0.6b-finetuned",
)
