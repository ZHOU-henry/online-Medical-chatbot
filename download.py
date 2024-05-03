from __future__ import annotations

from huggingface_hub import snapshot_download, hf_hub_download

def download_from_hf_hub(repo_id, local_dir, use_auth_token, filename=None):
    if filename is not None:
        res = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir, use_auth_token=use_auth_token, local_dir_use_symlinks=False)
    else:
        res = snapshot_download(repo_id=repo_id, local_dir=local_dir, use_auth_token=use_auth_token, local_dir_use_symlinks=False, ignore_patterns=["*.safetensors"])
    return res


if __name__ == "__main__":
    use_auth_token = "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" # replace "xxx" with your access token (see https://huggingface.co/docs/hub/security-tokens and https://huggingface.co/settings/tokens)

    ## repo_id = "microsoft/phi-2"
    ## local_dir = "C:/OneDrive/Desktop/HKU-DASC7606-A2-main/download/phi-1_5"

    repo_id = "meta-llama/Meta-Llama-3-8B" # "meta-llama/Meta-Llama-3-70B"
    local_dir = "C:/OneDrive/git/ARIN_7102/download/Llama-3-8B" # replace "xxx" with a real path and make sure that it has at least 3G of space

    download_from_hf_hub(repo_id=repo_id, use_auth_token=use_auth_token, local_dir=local_dir)

'''
    ## repo_id = "BAAI/bge-small-en-v1.5"
    ## local_dir = "xxx/bge-small-en-v1.5"

    repo_id = "BAAI/bge-small-en-v1.5"
    local_dir = "C:/OneDrive/git/ARIN_7102/download/bge-small-en-v1.5"

    download_from_hf_hub(repo_id=repo_id, use_auth_token=use_auth_token, local_dir=local_dir)
'''
