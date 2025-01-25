import os
import time
import h5py
import json
import cv2
from tqdm import tqdm
import glob

def play_images_with_filename_range(folder_path, start_frame=60, end_frame=150, delay=33):
    """
    播放指定范围内的图片，并在每帧上显示文件名。
    
    参数：
        folder_path (str): 包含图片的文件夹路径。
        start_frame (int): 起始帧号（包含）。
        end_frame (int): 结束帧号（包含）。
        delay (int): 每帧之间的延迟时间（毫秒），33 大约等于 30 FPS。
    """
    # 格式化帧号范围内的文件名
    file_names = [f"frame_{i:06}.jpg" for i in range(start_frame, end_frame + 1)]
    
    print(f"正在播放从 {start_frame} 到 {end_frame} 的图片...")
    
    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)
        # 读取图片
        image = cv2.imread(file_path)
        
        if image is None:
            print(f"无法读取图片: {file_name}")
            continue
        
        # 在图片上写上文件名
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (0, 255, 0)  # 绿色
        thickness = 2
        position = (50, 50)  # 文件名显示位置
        
        cv2.putText(image, file_name, position, font, font_scale, font_color, thickness, cv2.LINE_AA)
        
        # 显示图片
        cv2.imshow("Image Player", image)
        
        # 等待指定时间，按下 'q' 键退出
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()

def split_h5_file(input_file, output_file1, output_file2, x):
    """
    根据帧索引 x 将 HDF5 文件分为两个文件。
    """
    def recursive_copy(group, out_group1, out_group2, x):
        for key in group.keys():
            obj = group[key]
            if isinstance(obj, h5py.Group):  # 如果是组，递归调用
                out_group1.create_group(key)
                out_group2.create_group(key)
                recursive_copy(obj, out_group1[key], out_group2[key], x)
            elif isinstance(obj, h5py.Dataset):  # 如果是数据集，处理分割
                data = obj[:]
                if len(data.shape) > 0 and data.shape[0] > x:
                    out_group1.create_dataset(key, data=data[:x], dtype=obj.dtype)
                    out_group2.create_dataset(key, data=data[x:], dtype=obj.dtype)
                else:
                    out_group1.create_dataset(key, data=data, dtype=obj.dtype)
                    out_group2.create_dataset(key, data=data, dtype=obj.dtype)

    with h5py.File(input_file, 'r') as infile:
        with h5py.File(output_file1, 'w') as outfile1, h5py.File(output_file2, 'w') as outfile2:
            recursive_copy(infile, outfile1, outfile2, x)


def save_data(low_dim, image_paths, write_path, right_only):
    camera_names = ['cam_high', 'cam_left_wrist', 'cam_right_wrist'] \
    if not right_only else ['cam_high', 'cam_right_wrist']
    data_size = len(low_dim['action/arm/joint_position'])
    data_dict = {
        '/observations/qpos': [],
        '/action': [],
    }
    for cam_name in camera_names:
        data_dict[f'/observations/images/{cam_name}'] = []
    qpos = low_dim['observation/arm/joint_position']
    actions = low_dim['action/arm/joint_position']
    gripper_pos = low_dim['observation/eef/joint_position']
    gripper_actions = low_dim['action/eef/joint_position']
    for i in range(data_size):
        data_dict['/observations/qpos'].append(qpos[i][:6]+gripper_pos[i][0:1]+qpos[i][6:]+gripper_pos[i][1:2])
        data_dict['/action'].append(actions[i][:6]+gripper_actions[i][0:1]+actions[i][6:]+gripper_actions[i][1:2])
        for cam_name, img_path in zip(camera_names, image_paths):
            if os.path.exists(f"{img_path}/frame_{i:06}.png"):
                img = cv2.imencode(".jpg",cv2.imread(f"{img_path}/frame_{i:06}.png"), [int(cv2.IMWRITE_JPEG_QUALITY), 85])[1]
                img_flag = 'png'
            else:
                img = cv2.imread(f"{img_path}/frame_{i:06}.jpg")
                img_flag = 'jpg'
            data_dict[f'/observations/images/{cam_name}'].append(img)
    with h5py.File(write_path + '.hdf5', 'w', rdcc_nbytes=1024**2*2) as root:
        obs = root.create_group('observations')
        image = obs.create_group('images')
        for cam_name in camera_names:
            if img_flag == 'jpg':
                _ = image.create_dataset(cam_name, (data_size, 480, 640, 3), dtype='uint8',
                                            chunks=(1, 480, 640, 3), )
            elif img_flag == 'png':
                _ = image.create_dataset(cam_name, (data_size,), dtype='S40000', chunks=True)
        
        if not right_only:
            _ = obs.create_dataset('qpos', (data_size, 14))
            _ = root.create_dataset('action', (data_size, 14))
        else:
            _ = obs.create_dataset('qpos', (data_size, 7))
            _ = root.create_dataset('action', (data_size, 7))

        for name, array in data_dict.items():  
            root[name][...] = array
def main():
    right_only = True
    raw_dir = "/data/hpfs/Downloads/data/raw"
    name = "pick_place"
    sub_name_1 = 'pick'
    sub_name_2 = 'place'
    dataset_dir = "/data/hpfs/Downloads/data/dataset"
    episode_length = 250
    # os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(f"{dataset_dir}/{name}", exist_ok=True)
    os.makedirs(f"{dataset_dir}/{sub_name_1}", exist_ok=True)
    os.makedirs(f"{dataset_dir}/{sub_name_2}", exist_ok=True)
    raw_data_dirs = glob.glob(f"{raw_dir}/{name}_*")
    episode_output = 300
    sub_episode_output = 300
    episode_count = 300
    for raw_data_dir in raw_data_dirs:
        episodes_dir = os.listdir(raw_data_dir)
        episodes_dir = sorted([x for x in episodes_dir if x.isdigit()], key=lambda x: int(x))
        for episode in tqdm(episodes_dir):
            episode_dir = f"{raw_data_dir}/{episode}"
            imgs_dirs = os.listdir(episode_dir)
            imgs_dirs = sorted([f"{episode_dir}/{x}" for x in imgs_dirs if os.path.isdir(f"{episode_dir}/{x}")])
            with open(f"{episode_dir}/low_dim.json", 'r') as f:
                low_dim = json.load(f)
            episode_count += 1
            if len(low_dim['action/arm/joint_position']) < episode_length:
                continue
            else:
                save_data(low_dim, imgs_dirs, f"{dataset_dir}/{name}/{episode_output}", right_only=right_only)
                # play_images_with_filename_range(folder_path=imgs_dirs[1],
                #                                 start_frame=60,
                #                                 end_frame=150,
                #                                 delay=33)
                while True:
                    try:
                        x = int(input(f"请输入文件 {episode_dir} 的分割点 x（0-250）："))
                        if 0 <= x <= 250:
                            break
                        else:
                            print("分割点 x 必须在 0 和 250 之间，请重新输入。")
                    except ValueError:
                        print("输入无效，请输入一个整数。")

                split_h5_file(input_file=f"{dataset_dir}/{name}/{episode_output}.hdf5",
                              output_file1=f"{dataset_dir}/{sub_name_1}/{sub_episode_output}.hdf5",
                              output_file2= f"{dataset_dir}/{sub_name_2}/{sub_episode_output}.hdf5",
                              x=x)
                sub_episode_output += 1
                episode_output += 1
    print(f"Total episodes: {episode_count}")
    print(f"Total episodes saved: {episode_output}") 
    instruction_dict = {
        "instruction": "Pick up the block on the table and place it in the red square area.",
        "simplified_instruction": [
            "Pick up the block and put it in the red square.",
            "Lift the block from the table and place it in the red square.",
            "Grab the block on the table and drop it into the red square.",
        ],
        "expanded_instruction": [
            "Locate the block on the table, pick it up carefully, and gently move it to the red square area. Make sure the block is fully within the boundaries of the red square before releasing it.",
            "Find the block resting on the table, lift it securely, and carry it to the red square area. Place it down in a way that ensures the entire block is contained within the red square.",
            "Identify the block on the table, pick it up with a firm grip, and transfer it to the red square region. Position it so that it remains completely inside the red square, verifying it does not extend beyond the edges.",
        ],
    }

    sub_instruction_dict_1 = {
        "instruction": "Pick up the block on the table.",
        "simplified_instruction": [
            "Pick up the block from the table.",
            "Grab the block off the table.",
            "Lift the block from the table surface.",
        ],
        "expanded_instruction": [
            "Locate the block on the table, firmly grasp it without damaging it, and lift it so that it is completely off the table.",
            "Identify the block on the table, carefully secure it in your grip, and raise it from the surface, making sure you hold it steadily.",
            "Spot the block on the table, gently wrap your fingers around it, and lift it upwards until it is clear of the tabletop.",
        ],
    }

    sub_instruction_dict_2 = {
        "instruction": "Place the block in the red square area.",
        "simplified_instruction": [
            "Put the block inside the red square.",
            "Place the block into the red area.",
            "Lay the block down in the red square.",
        ],
        "expanded_instruction": [
            "Move the block carefully to the red square area and set it down, ensuring the entire block is contained within the boundary.",
            "Carry the block into the red square and gently lower it, making sure it stays entirely within the red lines.",
            "Transfer the block to the designated red region and position it so that it remains completely within the square, avoiding any overlap with the boundary.",
        ],
    }

    with open(f"{dataset_dir}/{name}/expanded_instruction_gpt-4-turbo.json", 'w') as f:
        json.dump(instruction_dict, f)

    with open(f"{dataset_dir}/{sub_name_1}/expanded_instruction_gpt-4-turbo.json", 'w') as f:
        json.dump(sub_instruction_dict_1, f)

    with open(f"{dataset_dir}/{sub_name_2}/expanded_instruction_gpt-4-turbo.json", 'w') as f:
        json.dump(sub_instruction_dict_2, f)
if __name__ == "__main__":
    main()
