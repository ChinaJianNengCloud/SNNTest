import os


def mk_dir(dir_path: str) -> bool:
    """
    没有就创建这个文件夹，有就直接返回True
    """
    # 为了防止是WindowsPath而报错，先转换一下
    path = str(dir_path).strip()
    if not os.path.exists(path) or not os.path.isdir(path):
        try:
            os.makedirs(path)
            # os.mkdir(path)
        except Exception as e:
            print(str(e))
            return False
    return True
