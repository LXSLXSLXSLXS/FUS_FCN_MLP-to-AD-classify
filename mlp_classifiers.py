from model_wrapper import MLP_Wrapper_A, MLP_Wrapper_B, MLP_Wrapper_C, MLP_Wrapper_D, MLP_Wrapper_E, MLP_Wrapper_F
from utils import read_json
import numpy as np

def mlp_A_train(exp_idx, repe_time, accu, config):
    # mlp model build on features selected from disease probability maps (DPMs) generated from FCN
    mlp_setting = config
    for i in range(repe_time):
        mlp = MLP_Wrapper_A(imbalan_ratio=mlp_setting['imbalan_ratio'],
                            fil_num=mlp_setting['fil_num'],
                            drop_rate=mlp_setting['drop_rate'],
                            batch_size=mlp_setting['batch_size'],
                            balanced=mlp_setting['balanced'],
                            roi_threshold=mlp_setting['roi_threshold'],
                            roi_count=mlp_setting['roi_count'],
                            choice=mlp_setting['choice'],
                            exp_idx=exp_idx,
                            seed=seed+i,
                            model_name='mlp_A',
                            metric='accuracy')
        mlp.train(lr=mlp_setting['learning_rate'],
                  epochs=mlp_setting['train_epochs'])
        accu_test = mlp.test(i)[2:]
        accu['A']['test'].append(accu_test)


def mlp_B_train(exp_idx, repe_time, accu, config):
    # mlp build on non-imaging features, including age, gender, MMSE
    mlp_setting = config
    for i in range(repe_time):
        mlp = MLP_Wrapper_B(imbalan_ratio=mlp_setting['imbalan_ratio'],
                            fil_num=mlp_setting['fil_num'],
                            drop_rate=mlp_setting['drop_rate'],
                            batch_size=mlp_setting['batch_size'],
                            balanced=mlp_setting['balanced'],
                            roi_threshold=mlp_setting['roi_threshold'],
                            roi_count=mlp_setting['roi_count'],
                            choice=mlp_setting['choice'],
                            exp_idx=exp_idx,
                            seed=seed+i,
                            model_name='mlp_B_BN',
                            metric='accuracy')
        mlp.train(lr=mlp_setting['learning_rate'],
                  epochs=mlp_setting['train_epochs'])
        accu_test= mlp.test(i)[2:]
        accu['B']['test'].append(accu_test)


def mlp_C_train(exp_idx, repe_time, accu, config):
    # mlp build on combined features of mlp_A and mlp_B
    mlp_setting = config
    for i in range(repe_time):
        mlp = MLP_Wrapper_C(imbalan_ratio=mlp_setting['imbalan_ratio'],
                            fil_num=mlp_setting['fil_num'],
                            drop_rate=mlp_setting['drop_rate'],
                            batch_size=mlp_setting['batch_size'],
                            balanced=mlp_setting['balanced'],
                            roi_threshold=mlp_setting['roi_threshold'],
                            roi_count=mlp_setting['roi_count'],
                            choice=mlp_setting['choice'],
                            exp_idx=exp_idx,
                            seed=seed+i,
                            model_name='mlp_C_BN',
                            metric='accuracy')
        mlp.train(lr=mlp_setting['learning_rate'],
                  epochs=mlp_setting['train_epochs'])
        accu_test = mlp.test(i)[2:]
        accu['C']['test'].append(accu_test)




def mlp_A(config):
    print('##################################################')
    print(config)
    accu = {'A':{'test':[]}}
    for exp_idx in range(repe_time):
        mlp_A_train(exp_idx, 3, accu, config)
    print('ADNI test accuracy ',
          'A {0:.4f}+/-{1:.4f}'.format(float(np.mean(accu['A']['test'])), float(np.std(accu['A']['test']))))


def mlp_B(config):
    print('##################################################')
    print(config)
    accu = {'B':{'test':[]}}
    for exp_idx in range(repe_time):
        mlp_B_train(exp_idx, 3, accu, config)
    print('ADNI test accuracy ',
          'B {0:.4f}+/-{1:.4f}'.format(float(np.mean(accu['B']['test'])), float(np.std(accu['B']['test']))))


def mlp_C(config):
    print('##################################################')
    print(config)
    accu = {'C':{'test':[]}}
    for exp_idx in range(repe_time):
        mlp_C_train(exp_idx, 3, accu, config)
    print('ADNI test accuracy ',
          'C {0:.4f}+/-{1:.4f}'.format(float(np.mean(accu['C']['test'])), float(np.std(accu['C']['test']))))



if __name__ == "__main__":
    config = read_json('./config.json')
    seed, repe_time = 1000, config['repeat_time']
    mlp_A(config["mlp_A"])
    mlp_B(config["mlp_B"])
    mlp_C(config["mlp_C"])

    