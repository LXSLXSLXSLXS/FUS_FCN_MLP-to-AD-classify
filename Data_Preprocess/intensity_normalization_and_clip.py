import numpy as np
import nibabel as nib
import sys
from glob import glob 

def nifti_to_numpy(file): #将 nifti 转换为 numpy
    data = nib.load(file).get_data()[:181, :217, :181]
    return data

def normalization(scan): #执行 z-score 体素归一化
    scan = (scan - np.mean(scan)) / np.std(scan)
    return scan

def clip(scan): #剪掉强度异常值（voxel<-1 or voxel>2.5）
    return np.clip(scan, -1, 2.5)

if __name__ == "__main__":
    folder = '../data/registration/raw_data/processed_data' #配准后的数据
    for file in glob(folder + '*.nii'):
        data = nifti_to_numpy(file)
        data = normalization(data)
        data = clip(data)
        np.save(file.replace('.nii', '.npy'), data)
    


