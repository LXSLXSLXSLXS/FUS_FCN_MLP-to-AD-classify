from utils import read_json, data_split
from model_wrapper import CNN_Wrapper, FCN_Wrapper
import torch
torch.backends.cudnn.benchmark = True


def cnn_main(seed):
    cnn_setting = config['cnn'] #从config.jason文件中读取CNN参数
    for exp_idx in range(repe_time):
        cnn = CNN_Wrapper(fil_num         = cnn_setting['fil_num'],
                          drop_rate       = cnn_setting['drop_rate'],
                          batch_size      = cnn_setting['batch_size'],
                          balanced        = cnn_setting['balanced'],
                          Data_dir        = cnn_setting['Data_dir'],
                          exp_idx         = exp_idx,
                          seed            = seed,
                          model_name      = 'cnn',
                          metric          = 'accuracy') #选择准确率作为评价指标
        cnn.train(lr     = cnn_setting['learning_rate'], #训练
                  epochs = cnn_setting['train_epochs'])
        cnn.test() #测试
        cnn.gen_features() #生成概率图


def fcn_main(seed):
    fcn_setting = config['fcn'] ##从config.jason文件中读取FCN参数
    for exp_idx in range(repe_time):
        fcn = FCN_Wrapper(fil_num        = fcn_setting['fil_num'],
                        drop_rate       = fcn_setting['drop_rate'],
                        batch_size      = fcn_setting['batch_size'],
                        balanced        = fcn_setting['balanced'],
                        Data_dir        = fcn_setting['Data_dir'],
                        patch_size      = fcn_setting['patch_size'],
                        exp_idx         = exp_idx,
                        seed            = seed,
                        model_name      = 'fcn',
                        metric          = 'accuracy') ##选择准确率作为评价指标
        fcn.train(lr     = fcn_setting['learning_rate'], #训练
                  epochs = fcn_setting['train_epochs'])
        fcn.test_and_generate_DPMs() #测试和生成概率图


if __name__ == "__main__":

    config = read_json('./config.json') #读取配置文件，获得模型的初始参数
    seed, repe_time = 1000, config['repeat_time']  # 设置模型的随机数和训练次数
    # 根据repe_time将数据集划分为训练集、验证集、测试集=6：2：2，划分repe_time次
    data_split(repe_time=repe_time)

    # 训练FCN #####################################
    with torch.cuda.device(2):  # specify which gpu to use
        fcn_main(seed)  # each FCN model will be independently trained on the corresponding data split

    # 训练CNN #####################################
    with torch.cuda.device(2): # specify which gpu to use
        cnn_main(seed)  # each CNN model will be independently trained on the corresponding data split
        



