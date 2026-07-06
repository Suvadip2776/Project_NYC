from huggingface_hub import HfApi
api=HfApi(); repo="sdananya/qwen3.6-27b-nycc"
api.upload_folder(folder_path="/workspace/models/qwen3.6-27b-nycc-final", repo_id=repo, path_in_repo="merged-final", repo_type="model")
print("FINAL_MODEL_UPLOADED")
