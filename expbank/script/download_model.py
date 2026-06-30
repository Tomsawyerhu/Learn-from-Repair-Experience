import os
import argparse
import sys
from pathlib import Path

def download_from_huggingface(repo_id, local_dir, use_mirror=True, token=None):
    """
    Download model from Hugging Face.
    """
    if use_mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        print(f"Using HF Mirror: {os.environ['HF_ENDPOINT']}")
    
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Error: 'huggingface_hub' is not installed. Please install it with: pip install huggingface_hub")
        sys.exit(1)
        
    print(f"Downloading {repo_id} from Hugging Face to {local_dir}...")
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
            token=token
        )
        print(f"Successfully downloaded {repo_id} to {local_dir}")
    except Exception as e:
        print(f"Error downloading from Hugging Face: {e}")
        sys.exit(1)

def download_from_modelscope(repo_id, local_dir):
    """
    Download model from ModelScope.
    """
    try:
        from modelscope.hub.snapshot_download import snapshot_download
    except ImportError:
        print("Error: 'modelscope' is not installed. Please install it with: pip install modelscope")
        sys.exit(1)
        
    print(f"Downloading {repo_id} from ModelScope to {local_dir}...")
    try:
        snapshot_download(
            model_id=repo_id,
            cache_dir=local_dir,
        )
        # ModelScope downloads to a subdirectory in cache_dir, we might want to handle that or just leave it.
        # snapshot_download in modelscope returns the path to the downloaded model.
        print(f"Successfully downloaded {repo_id} to {local_dir}")
    except Exception as e:
        print(f"Error downloading from ModelScope: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Generic Model Download Script (Hugging Face / ModelScope)")
    
    parser.add_argument("repo_id", default="BAAI/bge-large-en-v1.5", help="The repository ID of the model (e.g., BAAI/bge-large-en-v1.5)")
    parser.add_argument("--local-dir", default="./models", "-d", required=True, help="Local directory to save the model")
    parser.add_argument("--source", "-s", choices=["huggingface", "modelscope"], default="huggingface", help="Download source (default: huggingface)")
    parser.add_argument("--no-mirror", action="store_true", help="Disable HF mirror (use official Hugging Face endpoint)")
    parser.add_argument("--token", "-t", help="Hugging Face access token (optional)")
    
    args = parser.parse_args()
    
    # Ensure local directory exists
    local_path = Path(args.local_dir)
    local_path.mkdir(parents=True, exist_ok=True)
    
    if args.source == "huggingface":
        download_from_huggingface(
            repo_id=args.repo_id, 
            local_dir=str(local_path), 
            use_mirror=not args.no_mirror,
            token=args.token
        )
    elif args.source == "modelscope":
        download_from_modelscope(
            repo_id=args.repo_id, 
            local_dir=str(local_path)
        )

if __name__ == "__main__":
    main()
