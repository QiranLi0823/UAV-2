import os
import numpy as np
from tqdm import tqdm
from joblib import Parallel , delayed
from numpy.lib.format import open_memmap
from ThirdOrderRep import getThridOrderRep


def gen_angle_data_one_num_worker2(train_x, train_y, test_x, test_y):
    new_train_x = open_memmap('new_train_x.npy',dtype='float32',mode='w+',shape=(16431, 9, 300, 17, 2))
    new_test_x = open_memmap('new_test_x.npy',dtype='float32',mode='w+',shape=(4599, 9, 300, 17, 2))

    train_x = np.load(train_x,mmap_mode='r')
    train_y = np.load(train_y,mmap_mode='r')
    test_x = np.load(test_x,mmap_mode='r')
    test_y = np.load(test_y,mmap_mode='r')

    # N_train, T_train, _ = train_x.shape
    # N_test, T_test, _ = test_x.shape
    
    # train_x = train_x.reshape((16431, 300, 2, 17, 3)).transpose(0, 4, 1, 3, 2)
    # test_x = test_x.reshape((2000, 300, 2, 17, 3)).transpose(0, 4, 1, 3, 2)
    
    # print("shape:{}".format(train_x.shape))
    # print("shape:{}".format(test_x.shape))
    # exit()
    N_train = train_x.shape[0]
    N_test = test_x.shape[0]

    Parallel(n_jobs=8)(delayed(lambda i: new_train_x.__setitem__(i,getThridOrderRep(train_x[i])))(i) for i in tqdm(range(16431)))
    Parallel(n_jobs=8)(delayed(lambda i: new_test_x.__setitem__(i,getThridOrderRep(test_x[i])))(i) for i in tqdm(range(4599)))
    
    new_train_x = new_train_x.transpose(0, 1, 3, 2, 4)
    new_test_x = new_test_x.transpose(0, 1, 3, 2, 4)
    
    print("new_train_x shape:{}".format(new_train_x.shape))
    print("new_test_x shape:{}".format(new_test_x.shape))

    np.save('../../../data/train_angle.npy', new_train_x)
    np.save('../../../data/test_B_angle.npy', new_test_x)

if __name__ == '__main__':
    # gen_angle_data_one_num_worker('MMVRAC_CSv1.npz')
    # gen_angle_data_one_num_worker('MMVRAC_CSv2.npz')
    train_x = '../../../data/train_joint.npy'
    train_y = '../../../data/train_label.npy'
    test_x = '../../../data/test_joint_B.npy'
    test_y = '../../../data/test_B_label.npy'
    
    gen_angle_data_one_num_worker2(train_x, train_y, test_x, test_y)
    