import argparse
parser = argparse.ArgumentParser()

#通用参数
parser.add_argument("-d","--dataset_name",default='CAIL',type=str,choices=['CAIL'])  #数据集名称，与README.md和config.py对应

parser.add_argument("-up","--use_preprocessed",action='store_true')  #是否使用预处理后存储在本地的数据（路径由config.py指定）

parser.add_argument("-a","--analyse",action="store_true")  #是否打印对数据集的分析内容

parser.add_argument("-dp","--do_preprocess",default=None,choices=[None,'use_preprocessed','default'])  #对数据进行预处理工作

parser.add_argument("-ps","--preprocess_store",action='store_true')  
#储存数据预处理后的结果到config.py指定路径的文件夹中，方便下次使用。如do_preprocess=None将忽略该参数

parser.add_argument("-m","--model",default=None)  #使用的模型。如置None则为不运行模型（仅做数据分析和预处理等）

parser.add_argument('-dv','--gpu_device',default='cuda:0')  #这个只要是torch.device()可以接受的参数就行了

parser.add_argument('-b','--batch_size',default=128,type=int)

parser.add_argument('-o','--optimizer',default='Adam')

parser.add_argument('-l','--learning_rate',default=0.001,type=float)

#不固定的参数
parser.add_argument('-oa','--other_arguments',nargs='*')

args = parser.parse_args()
arg_dict=args.__dict__

dataset_name=arg_dict['dataset_name']
isAnalyse=arg_dict['analyse']

import sys,os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import config
from torch_ljp.dataset_utils import cail_analyse

if isAnalyse:
    if dataset_name=='CAIL':
        cail_analyse(data_path=config.cail_original_path,accu_path=config.cail_accu_path,law_path=config.cail_law_path)

