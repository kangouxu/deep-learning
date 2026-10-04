import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader

import IPython
e = IPython.embed  #调试断点
#处理数据集 区分仿真与真机
class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self,episode_ids,dataset_dir,camera_names,norm_stats):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids  #可用轨迹编号列表
        self.dataset_dir = dataset_dir  #数据集地址
        self.camera_names = camera_names  #相机名称
        self.norm_stats = norm_stats  #qpos、action的均值方差，用来标准化
        self.is_sim = None  #先占位  不知道是仿真还是真机
        self.__getitem__(0)  #读取第一条数据，就知道是不是仿真环境了

    def __len__(self):
        return len(self.episode_ids)  #返回总共有多少条数据

    def __getitem__(self, index):  #读取第n条数据看看是什么情况（仿真/真机）
        sample_full_episode = False  #如果true的话只会从第0步开始，一个episode只产生一个样本，false的话可以从不同位置开始，生成多条数据
        episode_id = self.episode_ids[index]  #从index读取编号id
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')  #拼接地址
        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']  #attrs 文件属性
            original_action_shape = root['/action'].shape
            episode_len = original_action_shape[0]
            if sample_full_episode:
                start_ts = 0  #从0开始
            else:
                start_ts = np.random.choice(episode_len)  #从整段长度中随机选择一个点
            qpos = root['/observations/qpos'][start_ts]  #这个时刻的关节角度
            qvel = root['/observations/qvel'][start_ts] #这个时刻的关节速度 
            image_dict = dict()  #相机名称列表
            for cam_name in self.camera_names:  #对于所有的相机
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts] #取当前这一帧的画面
            if is_sim:
                action = root['/action'][start_ts:]
                action_len = episode_len - start_ts  #动作有效长度 = 总步长 - 这个时间点
            else:
                action = root['/action'][max(0, start_ts - 1):] # 对齐时间戳 往前移一帧（真机延迟）
                action_len = episode_len - max(0, start_ts - 1) 
        self.is_sim = is_sim
        padded_action = np.zeros(original_action_shape, dtype=np.float32) #动作序列，无效action是0
        padded_action[:action_len] = action  #
        is_pad = np.zeros(episode_len)
        is_pad[action_len:] = 1  #剩下的全部填1（在padding中不会被考虑计算 loss）

        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])  #相机图片堆叠
        all_cam_images = np.stack(all_cam_images, axis=0)

        image_data = torch.from_numpy(all_cam_images)  #图像tensor
        qpos_data = torch.from_numpy(qpos).float()   #关节位置
        action_data = torch.from_numpy(padded_action).float()  #padding后的动作
        is_pad = torch.from_numpy(is_pad).bool()  

        #图像通道格式转换 （hdf5：k，h，w，c   cnn：k，c，h，w）
        image_data = torch.einsum('k h w c -> k c h w', image_data) 

        #归一化
        image_data = image_data / 255.0 #图像像素归一化  更适合神经网络计算和训练
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        return image_data, qpos_data, action_data, is_pad

#算 均值 和 标准差  用来归一化  
def get_norm_stats(dataset_dir, num_episodes):  #一个数据集地址，一个数量
    all_qpos_data = []
    all_action_data = []
    #读取位置、速度、动作信息
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')  #类似于上面的
        with h5py.File(dataset_path, 'r') as root:
            qpos = root['/observations/qpos'][()]
            qvel = root['/observations/qvel'][()]
            action = root['/action'][()]
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))

        #动作的均值 标准差
        action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)  #mean 均值
        action_std = all_action_data.std(dim=[0, 1], keepdim=True)    #std 标准差
        action_std = torch.clip(action_std, 1e-2, np.inf) # clipping  限制上下界 

        #位置的均值 标准差
        qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
        qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
        qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping


        #squeeze 删除维度为一的张量，如 （1，2，3）-》（2，3） 
        stats = {"action_mean": action_mean.numpy().squeeze(),   
                 "action_std": action_std.numpy().squeeze(),
                "qpos_mean": qpos_mean.numpy().squeeze(), 
                "qpos_std": qpos_std.numpy().squeeze(),
                "example_qpos": qpos}  
        
        return stats


def load_data(dataset_dir,num_episodes,camera_names,batch_size_train,batch_size_val):  #加载数据  划分训练集、测试集
    print(f'\nData from:{dataset_dir}\n')
    #划分测试集、训练集
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)#打乱数据集，防止都是同一类任务
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]  #按比例 前80%是训练集
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    norm_stats = get_norm_stats(dataset_dir, num_episodes)  #获取 均值 标准差  返回一个数组

    #构建数据集和验证集
    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats)
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats)
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=batch_size_train,   #训练批次 类似于cnn一次给32个图片
                                  shuffle=True, #打乱
                                  pin_memory=True, #锁页内存 搬到GPU更快
                                  num_workers=1, #子进程数量
                                  prefetch_factor=1)  #预加载 1 个 batch 的数据
    val_dataloader = DataLoader(val_dataset,
                                batch_size=batch_size_val, 
                                shuffle=True,  
                                pin_memory=True,
                                num_workers=1,
                                prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim

#仿真环境工具
#放方块
def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])  #不旋转
    return np.concatenate([cube_position, cube_quat])

#插销
def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

#一些工具
def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result   #计算均值loss

def detach_dict(d):  #丢弃计算图，节省显存
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):  #固定种子，保证可复现
    torch.manual_seed(seed)
    np.random.seed(seed)

