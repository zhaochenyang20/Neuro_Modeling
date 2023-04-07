import scipy.io
import os
import sys
import numpy as np
import pandas as pd

class DataLoader:
    '''
    Function:
        Load data from ./dataset
    Note:
        N represents the number of neural cells, M represents the number of records at different times
    Elements:
        global_C(N, M): changes of calcium signals(0.1 seconds between adjacent elements)
            钙信号强度
        global_S(N, M): The peak time of calcium activity in neuronal cells, a number other than 0 means that the activity of the neuron reaches its peak at this time.
            S spike 超过阈值会放电，非负浮点数
        global_centers(N ,2): the space location of neural cells
            细胞的空间位置，id 和前述的 N 对应
        brain_region_id(N ,1): the number of brain regions where neurons are located
            细胞所在的大脑区域，id 和前述的 N 对应，范围不大的 int，在 30 ~ 100 之间
        brain_region_name(N ,1): the name of brain regions where neurons are located
            细胞所在的大脑区域名字
        *_org: elements before clipping(infer_results_1 and infer_results_2 have different M, above elements use min(M1,M2))
            一共有两组数据，第二组稍微舍弃了 1% 左右的数据
    '''

    def __init__(self):
        print("======= Loading data =======")
        dataset_path = os.path.join(sys.path[0], "dataset")
        self.infer_results_1 = self.load_mat(os.path.join(dataset_path, "infer_results_1.mat"))
        self.infer_results_2 = self.load_mat(os.path.join(dataset_path, "infer_results_2.mat"))
        self.infer_results_id_1 = self.load_mat(os.path.join(dataset_path, "infer_results_id_1.mat"))
        self.infer_results_id_2 = self.load_mat(os.path.join(dataset_path, "infer_results_id_2.mat"))
        self.brain_region_name_1 = self.load_excel(os.path.join(dataset_path, "Infer results 1/brain_region_name1.xlsx"))
        self.brain_region_name_2 = self.load_excel(os.path.join(dataset_path, "Infer results 2/brain_region_name2.xlsx"))

        cat = lambda x,y:np.concatenate((x,y),axis=0)
        self.global_C = cat(self.infer_results_1["global_C"], self.infer_results_2["global_C"][:,:14122])
        self.global_S = cat(self.infer_results_1["global_S"], self.infer_results_2["global_S"][:,:14122])
        self.global_centers = cat(self.infer_results_1["global_centers"], self.infer_results_2["global_centers"])
        self.brain_region_id = cat(self.infer_results_id_1["brain_region_id"], self.infer_results_id_2["brain_region_id"])
        self.brain_region_name = cat(self.brain_region_name_1, self.brain_region_name_2)

        self.global_C_org = [self.infer_results_1["global_C"], self.infer_results_2["global_C"]]
        self.global_S_org = [self.infer_results_1["global_S"], self.infer_results_2["global_S"]]
        self.global_centers_org = [self.infer_results_1["global_centers"], self.infer_results_2["global_centers"]]
        self.brain_region_id_org = [self.infer_results_id_1["brain_region_id"], self.infer_results_id_2["brain_region_id"]]
        self.brain_region_name_org = [self.brain_region_name_1, self.brain_region_name_2]

    def load_mat(self, path):
        print(f"Loading matlab file from {path}")
        mat = scipy.io.loadmat(path)
        return mat

    def load_excel(self, path):
        print(f"Loading excel file from {path}")
        dict = pd.read_excel(path)
        first_element = dict.keys()[0]
        return np.concatenate((np.array([first_element]).reshape(1,1), np.array(dict)),axis=0)

