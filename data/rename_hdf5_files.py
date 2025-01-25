import os

def rename_hdf5_files(folder_path, new_prefix="file_", start_index=1):
    """
    批量重命名文件夹中的 .hdf5 文件。
    
    参数：
        folder_path (str): 包含 .hdf5 文件的文件夹路径。
        new_prefix (str): 重命名后的文件名前缀，例如 'file_'。
        start_index (int): 重命名时的起始编号。
    """
    # 确保路径存在
    if not os.path.exists(folder_path):
        print(f"路径不存在：{folder_path}")
        return
    
    # 获取文件夹中的所有 .hdf5 文件并排序
    hdf5_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.hdf5')], key=lambda x: int(x.split('.')[0]))
    
    if not hdf5_files:
        print(f"文件夹中没有 .hdf5 文件：{folder_path}")
        return

    print(f"找到 {len(hdf5_files)} 个 .hdf5 文件，开始重命名...")
    
    for index, file_name in enumerate(hdf5_files, start=start_index):
        old_path = os.path.join(folder_path, file_name)
        new_name = f"{new_prefix}{str(index)}.hdf5"  # 使用 zfill 补齐 6 位编号
        new_path = os.path.join(folder_path, new_name)
        
        # 执行重命名
        os.rename(old_path, new_path)
        print(f"重命名: {file_name} -> {new_name}")
    
    print("重命名完成！")


if __name__ == "__main__":
    # 替换为你的 .hdf5 文件夹路径
    folder_path = "/data/hpfs/Downloads/data/dataset/place"
    rename_hdf5_files(folder_path, new_prefix="", start_index=0)