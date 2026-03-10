from huggingface_hub import snapshot_download

def bajar_modelo():
    snapshot_download(
        repo_id="coqui/XTTS-v2",
        local_dir="./models/xtts_v2",
        local_dir_use_symlinks=False
    )

    print("Modelo descargado en ./models/xtts_v2")
