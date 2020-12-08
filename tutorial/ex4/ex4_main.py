import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np 

import platform,psutil
import time,os
import pandas as pd


def train(type='single'):
    """use fake image for training speed test"""
    target = torch.LongTensor(BATCH_SIZE).random_(1000).cuda()
    criterion = nn.CrossEntropyLoss()
    benchmark = {}
    for model_type in MODEL_LIST.keys():
        for model_name in MODEL_LIST[model_type]:
            model = getattr(model_type, model_name)(pretrained=False)
            if gpu_count > 1:
                model = nn.DataParallel(model,device_ids=range(gpu_count))
            model=getattr(model,type)()
            model=model.to('cuda')
            durations = []
            print('Benchmarking Training {} precision type {} '.format(type,model_name))
            for step,img in enumerate(trainloader):                
                img=getattr(img,type)()
                torch.cuda.synchronize()
                start = time.time()
                model.zero_grad()
                prediction = model(img.to('cuda'))
                loss = criterion(prediction, target)
                loss.backward()
                torch.cuda.synchronize()
                end = time.time()
                if step >= WARM_UP:
                    durations.append((end - start)*1000)
            print(model_name,' model average train time : ',sum(durations)/len(durations),'ms')
            del model
            benchmark[model_name] = durations
    return benchmark


def inference(type='float'):
    benchmark = {}
    with torch.no_grad():
        for model_type in MODEL_LIST.keys():
            for model_name in MODEL_LIST[model_type]:
                model = getattr(model_type, model_name)(pretrained=False)
                if gpu_count > 1:
                    model = nn.DataParallel(model,device_ids=range(gpu_count))
                model=getattr(model,type)()
                model=model.to('cuda')
                model.eval()
                durations = []
                print('Benchmarking Inference {} precision type {} '.format(type,model_name))
                for step,img in enumerate(trainloader):
                    img=getattr(img,type)()
                    torch.cuda.synchronize()
                    start = time.time()
                    model(img.to('cuda'))
                    torch.cuda.synchronize()
                    end = time.time()
                    if step >= WARM_UP:
                        durations.append((end - start)*1000)
                print(model_name,' model average inference time : ',sum(durations)/len(durations),'ms')
                del model
                benchmark[model_name] = durations
    return benchmark


class RandomDataset(Dataset):

    def __init__(self,  length):
        self.len = length
        self.data = torch.randn( 3, 224, 224,length)

    def __getitem__(self, index):
        return self.data[:,:,:,index]
        
    def __len__(self):
        return self.len


torch.backends.cudnn.benchmark = True

# Uncomment the following line to get a list of all models within the ResNet family
# print(models.resnet.__all__)

MODEL_LIST = { models.resnet: ['resnext101_32x8d'] } 
precisions=["float","half"] # "double" will take a substantial amount of time!

device_name=str(torch.cuda.get_device_name(0))
BATCH_SIZE=64

gpu_count = torch.cuda.device_count()
WARM_UP = 5   # Num of warm up runs


NUM_TEST = 50   # Num of Test
trainloader = DataLoader(dataset=RandomDataset( BATCH_SIZE*(WARM_UP + NUM_TEST)),
                         batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, 
                         num_workers=2)



if __name__ == '__main__':
    folder_name='new_results'
    path=''
    device_name="".join((device_name, '_',str(gpu_count),'_gpus_'))
    system_configs=str(platform.uname())
    system_configs='\n'.join((system_configs,str(psutil.cpu_freq()),'cpu_count: '+str(psutil.cpu_count()),'memory_available: '+str(psutil.virtual_memory().available)))
    gpu_configs=[torch.cuda.device_count(),torch.version.cuda,torch.backends.cudnn.version(),torch.cuda.get_device_name(0)]
    gpu_configs=list(map(str,gpu_configs))
    temp=['Number of GPUs on current device : ','CUDA Version : ','Cudnn Version : ','Device Name : ']

    os.makedirs(folder_name, exist_ok=True)
    now = time.localtime()
    start_time=str("%04d/%02d/%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec))
    print('benchmark start : ',start_time)

    for idx,value in enumerate(zip(temp,gpu_configs)):
        gpu_configs[idx]=''.join(value)
        print(gpu_configs[idx])
    print(system_configs)

    with open(os.path.join(folder_name,"system_info.txt"), "w") as f:
        f.writelines('benchmark start : '+start_time+'\n')
        f.writelines('system_configs\n\n')
        f.writelines(system_configs)
        f.writelines('\ngpu_configs\n\n')
        f.writelines(s + '\n' for s in gpu_configs )

    
    for precision in precisions:
        train_result=train(precision)
        train_result_df = pd.DataFrame(train_result)
        path=''.join((folder_name,'/',device_name,"_",precision,'_model_train_benchmark.csv'))
        train_result_df.to_csv(path, index=False)

        inference_result=inference(precision)
        inference_result_df = pd.DataFrame(inference_result)
        path=''.join((folder_name,'/',device_name,"_",precision,'_model_inference_benchmark.csv'))
        inference_result_df.to_csv(path, index=False)

    now = time.localtime()
    end_time=str("%04d/%02d/%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec))
    print('benchmark end : ',end_time)
    with open(os.path.join(folder_name,"system_info.txt"), "a") as f:
        f.writelines('benchmark end : '+end_time+'\n')
