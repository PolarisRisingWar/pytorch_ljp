import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-d","--dataset_name",default='CAIL',type=str,choices=['CAIL'])  #数据集名称，与README.md中的对应
parser.add_argument("-a","--analyse",action="store_true")  #是否打印对数据集的分析内容

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

