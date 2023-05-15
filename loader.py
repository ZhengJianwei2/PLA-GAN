import os # 程序与操作系统进行交互的接口。
from glob import glob # 返回所有匹配的文件路径列表
import numpy as np
from torch.utils.data import Dataset, DataLoader  # PyTorch中数据读取
from data_aug import data_augmentation


class ct_dataset(Dataset):
    def __init__(self, mode, load_mode, saved_path, test_patient, patch_n=None, patch_size=None, transform=None):
        assert mode in ['train', 'test'], "mode is 'train' or 'test'"  #  测试这个condition，如果condition为false，那么raise一个AssertionError出来
        assert load_mode in [0,1], "load_mode is 0 or 1"

        input_path = sorted(glob(os.path.join(saved_path, '*_input.npy')))  # 查找符合要求的文件
        target_path = sorted(glob(os.path.join(saved_path, '*_target.npy')))
        self.load_mode = load_mode
        self.patch_n = patch_n
        self.patch_size = patch_size
        self.transform = transform
        
        if mode == 'train':   # read path-> load data
            input_ = [f for f in input_path if test_patient not in f]   ## do not use test patient for train
            target_ = [f for f in target_path if test_patient not in f]
            if load_mode == 0: # batch data load
                self.input_ = input_
                self.target_ = target_
            else: # all data load
                self.input_ = [np.load(f) for f in input_]
                self.target_ = [np.load(f) for f in target_]
        else: # mode =='test'
            input_ = [f for f in input_path if test_patient in f]
            target_ = [f for f in target_path if test_patient in f]
            if load_mode == 0:
                self.input_ = input_
                self.target_ = target_
            else:
                self.input_ = [np.load(f) for f in input_]
                self.target_ = [np.load(f) for f in target_]

    def __len__(self):
        return len(self.target_)

    def __getitem__(self, idx):  # 自带的索引取值方法
        input_img, target_img = self.input_[idx], self.target_[idx]
        if self.load_mode == 0:
            input_img, target_img = np.load(input_img), np.load(target_img)  # 读取npy文件

        if self.transform:
            input_img = self.transform(input_img)
            target_img = self.transform(target_img)

        if self.patch_size:
            input_patches, target_patches = get_patch(input_img,
                                                      target_img,
                                                      self.patch_n,
                                                      self.patch_size)
            return (input_patches, target_patches)
        else:
            return (input_img, target_img)

def get_patch(full_input_img, full_target_img, patch_n, patch_size):
    assert full_input_img.shape == full_target_img.shape
    patch_input_imgs = []
    patch_target_imgs = []
    h, w = full_input_img.shape
    new_h, new_w = patch_size, patch_size
    if patch_size == h:    ## if same as original image return original image
        return full_input_img, full_target_img
    for _ in range(patch_n//2):  #//2 1/2for half original kept, for data augmentation
        top = np.random.randint(0, h-new_h)  # 随机选择位置，取指定的patch
        left = np.random.randint(0, w-new_w)
        patch_input_img = full_input_img[top:top+new_h, left:left+new_w]
        patch_target_img = full_target_img[top:top+new_h, left:left+new_w]
        #print('patch shape:',patch_input_img.shape)
        
        ## original image send, keep the original image first
        patch_input_imgs.append(patch_input_img)  
        patch_target_imgs.append(patch_target_img)
        
        ## data augment 2/2
        tmp = np.random.randint(1,8)  # 对于一半patch保持，剩下一半进行一次数据增强，
        patch_input_img = data_augmentation(patch_input_img, tmp)
        patch_target_img = data_augmentation(patch_target_img, tmp)
        
        patch_input_imgs.append(patch_input_img)
        patch_target_imgs.append(patch_target_img)
        
    return np.array(patch_input_imgs), np.array(patch_target_imgs) # 列表不存在维度，数组是有维度的 ，因此要转化为数组


# =============================================================================
# def get_loader(mode='train', load_mode=0,
#                saved_path=None, test_patient='L506',
#                patch_n=None, patch_size=None,
#                transform=None, batch_size=32, num_workers=6):
#     dataset_ = ct_dataset(mode, load_mode, saved_path, test_patient, patch_n, patch_size, transform)
#     data_loader = DataLoader(dataset=dataset_, batch_size=batch_size, shuffle=True, num_workers=num_workers)
#     return data_loader
# =============================================================================

def get_loader(mode='train', load_mode=0,
               saved_path=None, test_patient='L506',
               patch_n=None, patch_size=None,
               transform=None, batch_size=32, shuffle=True,num_workers=6):
    dataset_ = ct_dataset(mode, load_mode, saved_path, test_patient, patch_n, patch_size, transform)
    data_loader = DataLoader(dataset=dataset_, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers) # 这里的输入是一个元组组成的数组dataset，但是上面return的是两个 
    return data_loader
# torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=None, sampler=None, batch_sampler=None, num_workers=0, 
# collate_fn=None, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None, multiprocessing_context=None, generator=None, 
# *, prefetch_factor=2, persistent_workers=False, pin_memory_device='')