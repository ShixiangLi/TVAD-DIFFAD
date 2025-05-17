import numpy as np
from PIL import Image
import os
import sys
import random # <--- 引入 random 模块

# --- 配置参数 ---
VIDEO_DATA_PATH = 'datasets/raw_data/video_data.npy'  # 图像数据 .npy 文件路径 (N, H, W, C)
LABEL_DATA_PATH = 'datasets/raw_data/labels.npy'      # 标签数据 .npy 文件路径 (N, H, W)

# --- 新的输出目录结构参数 ---
# 假设您的 "combustion_dataset/chamber" 结构在 OUTPUT_BASE_DIR 下创建
# 或者您直接指定顶级目录 combustion_dataset，然后在脚本内部创建 chamber 子目录
# 为了与截图匹配，我们假定 OUTPUT_BASE_DIR 指向 combustion_dataset/chamber
OUTPUT_BASE_DIR = 'datasets/combustion_dataset/chamber' # <--- 修改这里以匹配截图的顶级目录

# 测试集图像输出目录 (根据新结构调整)
TEST_GOOD_IMAGE_DIR = os.path.join(OUTPUT_BASE_DIR, 'test', 'good')
TEST_ANOMALY_IMAGE_DIR_PREFIX = os.path.join(OUTPUT_BASE_DIR, 'test') # 例如 test/burn, test/defect2

# 测试集真值掩码输出目录 (根据新结构调整)
TEST_GOOD_GT_DIR = os.path.join(OUTPUT_BASE_DIR, 'ground_truth', 'good')
TEST_ANOMALY_GT_DIR_PREFIX = os.path.join(OUTPUT_BASE_DIR, 'ground_truth') # 例如 ground_truth/burn

# 训练集图像输出目录 (根据新结构调整)
TRAIN_GOOD_IMAGE_DIR = os.path.join(OUTPUT_BASE_DIR, 'train', 'good')

FILENAME_PREFIX = 'frame'    # 输出文件名的前缀
NUM_TEST_NORMAL = 2000        # 测试集中正常样本数量
NUM_TEST_ABNORMAL = 2000      # 测试集中异常样本数量

# 假设我们只有一个异常类别，比如 "burn"，或者您需要一种方式来命名异常文件夹
# 如果您的 .npy 标签本身不区分类别，我们就统一放到一个 "anomaly" 文件夹下
# 如果您的截图中的 "burn" 是一个具体的异常类别名称，您需要在这里指定或从数据中推断
DEFAULT_ANOMALY_CLASS_NAME = 'anomaly' # 或者 'burn' 如果只有一个异常类型
# ----------------

def process_and_split_data_custom(video_path, label_path,
                                  train_good_img_dir,
                                  test_good_img_dir, test_anomaly_img_dir_base, # test_anomaly_img_dir_base -> test/
                                  test_good_gt_dir, test_anomaly_gt_dir_base,   # test_anomaly_gt_dir_base -> ground_truth/
                                  anomaly_class_name, # 用于创建如 test/burn, ground_truth/burn
                                  num_test_normal, num_test_abnormal,
                                  prefix):
    """
    加载图像(N, H, W, C)和标签(N, H, W)格式的.npy数据，
    按指定数量随机分割为训练/测试集，并保存为 PNG 文件到新目录结构。
    测试集中的正常样本也会生成全零掩码。
    """
    print(f"开始自定义处理与分割...")
    print(f"图像数据路径: {video_path}")
    print(f"标签数据路径: {label_path}")

    # --- 1. 加载数据 (与原脚本相同) ---
    try:
        print("正在加载图像数据...")
        video_data = np.load(video_path)
        print(f"图像数据加载成功，维度: {video_data.shape}")
    except FileNotFoundError:
        print(f"错误: 找不到图像数据文件 {video_path}")
        sys.exit(1)
    except Exception as e:
        print(f"加载图像数据时出错: {e}")
        sys.exit(1)

    try:
        print("正在加载标签数据...")
        label_data = np.load(label_path)
        print(f"标签数据加载成功，维度: {label_data.shape}")
    except FileNotFoundError:
        print(f"错误: 找不到标签数据文件 {label_path}")
        sys.exit(1)
    except Exception as e:
        print(f"加载标签数据时出错: {e}")
        sys.exit(1)

    # --- 2. 验证数据维度 (与原脚本相同) ---
    if video_data.ndim != 4:
        print(f"错误: 图像数据的维度必须为 4 (N, H, W, C)。实际维度: {video_data.ndim}")
        sys.exit(1)
    if label_data.ndim != 3:
        print(f"错误: 标签数据的维度必须为 3 (N, H, W)。实际维度: {label_data.ndim}")
        sys.exit(1)

    N_img, H_img, W_img, C_img = video_data.shape
    N_lbl, H_lbl, W_lbl = label_data.shape

    if N_img != N_lbl:
        print("错误: 图像数据和标签数据的样本数量 (N) 不匹配。")
        sys.exit(1)
    N = N_img

    if H_img != H_lbl or W_img != W_lbl:
        print("警告: 图像数据和标签数据的空间维度 (H, W) 不匹配。将使用图像维度。")
    H, W = H_img, W_img

    print(f"总样本数 (N): {N}, 图像尺寸: ({H}, {W}, {C_img})")

    # --- 3. 创建输出目录 ---
    # 注意：这里的 anomaly_class_name 用于构建具体的异常子目录
    actual_test_anomaly_img_dir = os.path.join(test_anomaly_img_dir_base, anomaly_class_name)
    actual_test_anomaly_gt_dir = os.path.join(test_anomaly_gt_dir_base, anomaly_class_name)

    os.makedirs(train_good_img_dir, exist_ok=True)
    os.makedirs(test_good_img_dir, exist_ok=True)
    os.makedirs(actual_test_anomaly_img_dir, exist_ok=True) # 例如 .../test/burn/
    os.makedirs(test_good_gt_dir, exist_ok=True)
    os.makedirs(actual_test_anomaly_gt_dir, exist_ok=True)   # 例如 .../ground_truth/burn/
    print(f"已确保输出目录存在。")


    # --- 4. 预分类样本索引 ---
    normal_indices = []
    abnormal_indices = []
    for i in range(N):
        label_map = label_data[i] # Shape: (H, W)
        if np.any(label_map != 0):
            abnormal_indices.append(i)
        else:
            normal_indices.append(i)

    print(f"找到 {len(normal_indices)} 个正常样本和 {len(abnormal_indices)} 个异常样本。")

    # --- 5. 随机抽样测试集 ---
    random.shuffle(normal_indices)
    random.shuffle(abnormal_indices)

    # 抽取测试集正常样本
    actual_num_test_normal = min(num_test_normal, len(normal_indices))
    test_normal_indices = normal_indices[:actual_num_test_normal]
    if actual_num_test_normal < num_test_normal:
        print(f"警告: 正常样本不足 {num_test_normal} 个，实际抽取 {actual_num_test_normal} 个作为测试集正常样本。")

    # 抽取测试集异常样本
    actual_num_test_abnormal = min(num_test_abnormal, len(abnormal_indices))
    test_abnormal_indices = abnormal_indices[:actual_num_test_abnormal]
    if actual_num_test_abnormal < num_test_abnormal:
        print(f"警告: 异常样本不足 {num_test_abnormal} 个，实际抽取 {actual_num_test_abnormal} 个作为测试集异常样本。")

    print(f"将抽取 {len(test_normal_indices)} 个正常样本和 {len(test_abnormal_indices)} 个异常样本作为测试集。")

    # --- 6. 确定训练集索引 ---
    # 训练集是所有未被选入测试集的正常样本
    # normal_indices 中前面 actual_num_test_normal 个是测试集，后面的是训练集
    train_normal_indices = normal_indices[actual_num_test_normal:]
    print(f"剩余 {len(train_normal_indices)} 个正常样本作为训练集。")

    # --- 7. 保存文件 ---
    num_digits = len(str(N - 1)) # 用于文件名格式化

    # 保存训练集正常样本
    print("\n正在保存训练集正常样本...")
    for i, original_idx in enumerate(train_normal_indices):
        img_slice = video_data[original_idx]
        # ... (与原脚本相同的图像保存逻辑: 确定模式, 转uint8, 保存到 train_good_img_dir) ...
        # <<<< 您需要从原脚本中复制粘贴图像保存逻辑到这里，并修改保存路径 >>>>
        base_filename = f"{prefix}_{original_idx:0{num_digits}d}.png" # 使用原始索引命名，保持唯一性
        image_save_path = os.path.join(train_good_img_dir, base_filename)
        
        # (图像处理和保存逻辑 - 开始)
        if C_img == 1: img_array = img_slice.squeeze(axis=-1); mode = 'L'
        elif C_img == 3: img_array = img_slice; mode = 'RGB'
        elif C_img == 4: img_array = img_slice; mode = 'RGBA'
        elif C_img == 0: print(f"错误: 样本 {original_idx} 通道数为0"); continue
        else: img_array = img_slice[:, :, 0]; mode = 'L'

        if img_array.dtype != np.uint8:
            if np.max(img_array) <= 1.0 and np.min(img_array) >= 0.0:
                img_array = (img_array * 255).astype(np.uint8)
            else:
                img_array = img_array.astype(np.uint8)
        try:
            pil_image = Image.fromarray(img_array, mode=mode)
            pil_image.save(image_save_path)
        except Exception as e:
            print(f"\n保存训练图像 {image_save_path} 时出错: {e}")
        # (图像处理和保存逻辑 - 结束)
        if (i + 1) % 100 == 0: print(f"\r已保存 {i+1}/{len(train_normal_indices)} 个训练样本", end="")
    print(f"\n训练集正常样本保存完毕。")

    # 保存测试集正常样本及其全零掩码
    print("\n正在保存测试集正常样本及其掩码...")
    for i, original_idx in enumerate(test_normal_indices):
        img_slice = video_data[original_idx]
        # ... (图像保存逻辑到 test_good_img_dir) ...
        # <<<< 复制粘贴图像保存逻辑，修改保存路径 >>>>
        base_filename = f"{prefix}_{original_idx:0{num_digits}d}.png"
        image_save_path = os.path.join(test_good_img_dir, base_filename)
        # (图像处理和保存逻辑 - 开始)
        if C_img == 1: img_array = img_slice.squeeze(axis=-1); mode = 'L'
        elif C_img == 3: img_array = img_slice; mode = 'RGB'
        elif C_img == 4: img_array = img_slice; mode = 'RGBA'
        elif C_img == 0: print(f"错误: 样本 {original_idx} 通道数为0"); continue
        else: img_array = img_slice[:, :, 0]; mode = 'L'

        if img_array.dtype != np.uint8:
            if np.max(img_array) <= 1.0 and np.min(img_array) >= 0.0:
                img_array = (img_array * 255).astype(np.uint8)
            else:
                img_array = img_array.astype(np.uint8)
        try:
            pil_image = Image.fromarray(img_array, mode=mode)
            pil_image.save(image_save_path)
        except Exception as e:
            print(f"\n保存测试(正常)图像 {image_save_path} 时出错: {e}")
        # (图像处理和保存逻辑 - 结束)

        # 保存全零掩码
        zero_mask = np.zeros((H, W), dtype=np.uint8)
        pil_mask = Image.fromarray(zero_mask, mode='L')
        mask_save_path = os.path.join(test_good_gt_dir, base_filename)
        try:
            pil_mask.save(mask_save_path)
        except Exception as e:
            print(f"\n保存测试(正常)掩码 {mask_save_path} 时出错: {e}")
        if (i + 1) % 100 == 0: print(f"\r已保存 {i+1}/{len(test_normal_indices)} 个测试集正常样本", end="")
    print(f"\n测试集正常样本及其掩码保存完毕。")

    # 保存测试集异常样本及其二值掩码
    print("\n正在保存测试集异常样本及其掩码...")
    for i, original_idx in enumerate(test_abnormal_indices):
        img_slice = video_data[original_idx]
        label_map = label_data[original_idx]
        # ... (图像保存逻辑到 actual_test_anomaly_img_dir) ...
        # <<<< 复制粘贴图像保存逻辑，修改保存路径 >>>>
        base_filename = f"{prefix}_{original_idx:0{num_digits}d}.png"
        image_save_path = os.path.join(actual_test_anomaly_img_dir, base_filename) # 例如 .../test/burn/frame_xxxx.png
         # (图像处理和保存逻辑 - 开始)
        if C_img == 1: img_array = img_slice.squeeze(axis=-1); mode = 'L'
        elif C_img == 3: img_array = img_slice; mode = 'RGB'
        elif C_img == 4: img_array = img_slice; mode = 'RGBA'
        elif C_img == 0: print(f"错误: 样本 {original_idx} 通道数为0"); continue
        else: img_array = img_slice[:, :, 0]; mode = 'L'

        if img_array.dtype != np.uint8:
            if np.max(img_array) <= 1.0 and np.min(img_array) >= 0.0:
                img_array = (img_array * 255).astype(np.uint8)
            else:
                img_array = img_array.astype(np.uint8)
        try:
            pil_image = Image.fromarray(img_array, mode=mode)
            pil_image.save(image_save_path)
        except Exception as e:
            print(f"\n保存测试(异常)图像 {image_save_path} 时出错: {e}")
        # (图像处理和保存逻辑 - 结束)

        # ... (与原脚本相同的二值掩码创建和保存逻辑，保存到 actual_test_anomaly_gt_dir) ...
        # <<<< 复制粘贴掩码创建和保存逻辑，修改保存路径 >>>>
        binary_mask = np.zeros_like(label_map, dtype=np.uint8)
        try:
            binary_mask[label_map != 0] = 255
        except TypeError: # 处理可能的类型问题
            try:
                non_zero_indices = label_map.astype(bool) if label_map.dtype == object else label_map != 0
                binary_mask[non_zero_indices] = 255
            except Exception as mask_err:
                print(f"\n错误: 无法为样本 {original_idx} 创建二值掩码: {mask_err}"); continue
        
        pil_mask = Image.fromarray(binary_mask, mode='L')
        mask_save_path = os.path.join(actual_test_anomaly_gt_dir, base_filename) # 例如 .../ground_truth/burn/frame_xxxx.png
        try:
            pil_mask.save(mask_save_path)
        except Exception as e:
            print(f"\n保存测试(异常)掩码 {mask_save_path} 时出错: {e}")

        if (i + 1) % 50 == 0: print(f"\r已保存 {i+1}/{len(test_abnormal_indices)} 个测试集异常样本", end="")
    print(f"\n测试集异常样本及其掩码保存完毕。")

    print(f"\n--- 自定义处理与分割完成！ ---")
    print(f"总样本数: {N}")
    print(f"训练集 (正常): {len(train_normal_indices)} (图像保存至 '{train_good_img_dir}')")
    print(f"测试集 (正常): {len(test_normal_indices)} (图像保存至 '{test_good_img_dir}', 掩码保存至 '{test_good_gt_dir}')")
    print(f"测试集 (异常 - '{anomaly_class_name}'): {len(test_abnormal_indices)} (图像保存至 '{actual_test_anomaly_img_dir}', 掩码保存至 '{actual_test_anomaly_gt_dir}')")


if __name__ == "__main__":
    # 调用修改后的函数
    # 您需要根据您的具体异常类别名称来设置 anomaly_class_name
    # 如果您的数据只有一个异常类型 "burn" 就像截图那样:
    anomaly_folder_name = "burn" # 或者您希望的任何名称，例如 'defects'

    process_and_split_data_custom(
        video_path=VIDEO_DATA_PATH,
        label_path=LABEL_DATA_PATH,
        train_good_img_dir=TRAIN_GOOD_IMAGE_DIR,
        test_good_img_dir=TEST_GOOD_IMAGE_DIR,
        test_anomaly_img_dir_base=TEST_ANOMALY_IMAGE_DIR_PREFIX, # test/
        test_good_gt_dir=TEST_GOOD_GT_DIR,
        test_anomaly_gt_dir_base=TEST_ANOMALY_GT_DIR_PREFIX,   # ground_truth/
        anomaly_class_name=anomaly_folder_name, # 这会创建 test/burn 和 ground_truth/burn
        num_test_normal=NUM_TEST_NORMAL,
        num_test_abnormal=NUM_TEST_ABNORMAL,
        prefix=FILENAME_PREFIX
    )