from typing import Sequence
import cv2
import json
import numpy as np
import os
import io
import random
import scipy.io
import sys
import torch.utils.data.dataset
from datetime import datetime as dt
from enum import Enum, unique
import sys
sys.path.insert(0, '../')
from data.imgio_gen import readgen
# import utils.network_utils

class DatasetType(Enum):
    TRAIN = 0
    TEST  = 1


class ImageProcessDataset(torch.utils.data.dataset.Dataset):    
    def __init__(self,dataset_type,DATASET_JSON_FILE_PATH=None,transforms = None):
        self.transforms = transforms
        # DATASET_JSON_FILE_PATH='/home/xuqian/transformer/COLA_Net/COLA-Net-Collaborative-Attention-Network-for-Image-Restoration-main/DN_Real/data/IPT_test.json'
        self.DATASET_JSON_FILE_PATH=DATASET_JSON_FILE_PATH 
        DATASET_ROOT = '/home/xuqian/dataset_GOPRO/'
        self.img_blur_path_template = os.path.join(DATASET_ROOT ,'%s/%s/input/%s.png')
        self.img_clear_path_template = os.path.join(DATASET_ROOT ,'%s/%s/GT/%s.png')
        with io.open(DATASET_JSON_FILE_PATH, encoding='utf-8') as file:
            self.files_list = json.loads(file.read())
        sequence=[]        
        sequences = []
        # Load data for each sequence

        for file in self.files_list:
            if dataset_type == DatasetType.TRAIN and file['phase'] == 'train':
                name = file['name']
                phase = file['phase']
                samples = file['sample']
                sam_len = len(samples)
                seq_len = 1
                seq_num = int(sam_len/seq_len)                
                for n in range(seq_num):                    
                    sequence = self.get_files_of_taxonomy(phase, name, samples[seq_len*n: seq_len*(n+1)])
                    sequences.extend(sequence)

                if not seq_len%seq_len == 0:
                    sequence = self.get_files_of_taxonomy(phase, name, samples[-seq_len:])
                    sequences.extend(sequence)
                    seq_num += 1
                print('[INFO] %s Collecting files of Taxonomy [Name = %s]' % (dt.now(), name + ': ' + str(seq_num)))


            elif dataset_type == DatasetType.TEST and file['phase'] == 'test':
                name = file['name']
                phase = file['phase']
                samples = file['sample']
                sam_len = len(samples)
                seq_len = 1
                seq_num = int(sam_len / seq_len)
                for n in range(seq_num):
                    sequence = self.get_files_of_taxonomy(phase, name, samples[seq_len*n: seq_len*(n+1)])
                    sequences.extend(sequence)

                if not seq_len % seq_len == 0:
                    sequence = self.get_files_of_taxonomy(phase, name, samples[-seq_len:])
                    sequences.extend(sequence)
                    seq_num += 1

                print('[INFO] %s Collecting files of Taxonomy [Name = %s]' % (dt.now(), name + ': ' + str(seq_num)))
        self.file_list=sequences

    def get_files_of_taxonomy(self, phase, name, samples):
        n_samples = len(samples)
        seq_blur_paths = []
        seq_clear_paths = []
        sequence = []

        for sample_idx, sample_name in enumerate(samples):
            # Get file path of img
            img_blur_path = self.img_blur_path_template % (phase, name, sample_name)
            img_clear_path = self.img_clear_path_template % (phase, name, sample_name)
            # print(img_blur_path)
            # print(img_clear_path)
            if os.path.exists(img_blur_path) and os.path.exists(img_clear_path):
                seq_blur_paths.append(img_blur_path)
                seq_clear_paths.append(img_clear_path)

        if not seq_blur_paths == [] and not seq_clear_paths == []:
            sequence.append({
                'name': name,
                'phase': phase,
                'length': n_samples,
                'seq_blur': seq_blur_paths,
                'seq_clear': seq_clear_paths,
            })
        return sequence

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        name, seq_blur, seq_clear = self.get_datum(idx)
        seq_blur, seq_clear = self.transforms(seq_blur, seq_clear)
        sample={"name":name,"blur":seq_blur,"clear":seq_clear}
        return  sample

    def get_datum(self, idx):
        name = self.file_list[idx]['name']
        phase = self.file_list[idx]['phase']
        length = self.file_list[idx]['length']
        seq_blur_paths = self.file_list[idx]['seq_blur']
        seq_clear_paths = self.file_list[idx]['seq_clear']
        seq_blur = []
        seq_clear = []
        # print(length)
        for i in range(length):
            # print('path1',seq_blur_paths[i])
            # print('path2',seq_clear_paths[i])
            img_blur = readgen(seq_blur_paths[i]).astype(np.float32)
            img_clear = readgen(seq_clear_paths[i]).astype(np.float32)
            seq_blur.append(img_blur)
            seq_clear.append(img_clear)
        
        if phase == 'train' and random.random() < 0.5:
            # random reverse
            seq_blur.reverse()
            seq_clear.reverse()
       
        return name, seq_blur, seq_clear

# if __name__=='__main__':
#     blurdataset=ImageProcessDataset(DatasetType.TEST)
#     blurdataloader=torch.utils.data.DataLoader(blurdataset,batch_size=1, shuffle=True, num_workers=0)
#     print("OKK")


