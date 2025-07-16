from huggingface_hub import HfApi, upload_folder

api = HfApi(token='')
# repo_id = api.create_repo(                # 若已存在会抛错，可加 exist_ok=True
#     repo_id="Franklin2002/procgen-dataset",
#     repo_type="dataset",
#     private=True,
#     token='')                         # 关键参数
api.upload_large_folder(
    folder_path="./../data/procgen",
    repo_id="Franklin2002/procgen-dataset",
    repo_type="dataset",
    )                     # 全部文件上传到根目录
