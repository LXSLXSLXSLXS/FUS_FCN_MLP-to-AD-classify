import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import os
import sys

def back_remove(file, temp, new_path): #背景去除

    if not os.path.exists(new_path):
        os.mkdir(new_path)

    data = np.load(file)
    new_data = data[:,:,:]

    stack = [(0, 0, 0), (180, 0, 0), (0, 216, 0), (180, 216, 0)]
    visited = set([(0, 0, 0), (180, 0, 0), (0, 216, 0), (180, 216, 0)])

    def valid(x, y, z):
        if x < 0 or x >= 181:
            return False
        if y < 0 or y >= 217:
            return False
        if z < 0 or z >= 181:
            return False
        return True

    while stack:
        x, y, z = stack.pop()
        for dx, dy, dz in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]:
            new_x, new_y, new_z = x + dx, y + dy, z + dz
            if valid(new_x, new_y, new_z) and (new_x, new_y, new_z) not in visited \
            and data[new_x, new_y, new_z] < -0.6 and temp[new_x, new_y, new_z] < 0.8:
                visited.add((new_x, new_y, new_z))
                new_data[new_x, new_y, new_z] = -10
                stack.append((new_x, new_y, new_z))

    filename = file.split('/')[-1]
    plt.subplot(131)
    plt.imshow(new_data[100, :, :])
    plt.subplot(132)
    plt.imshow(new_data[:, 100, :])
    plt.subplot(133)
    plt.imshow(new_data[:, :, 100])
    plt.savefig(new_path + filename.replace('npy', 'jpg')) #将背景去除的图片保存为jpg，便于观察
    plt.close()
    
    new_data = np.where(new_data==-10, -np.ones((181, 217, 181)), new_data).astype(np.float32)
    np.save(new_path + filename, new_data)

if __name__ == "__main__":
    folder = '../data/registration/raw_data/processed_data' #进行体素归一化和剪去异常值后的数据位置
    out_folder = '../data/back_remove' #背景移除后的数据存放位置
    temp = np.load('./brain_region.nii') #加载归一化后的.nii数据
    for file in glob(folder + '*.nii'):
        back_remove(file, temp, out_folder) #依次按照文件名进行背景除去操作




    

    