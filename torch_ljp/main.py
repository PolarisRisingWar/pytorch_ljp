import argparse
parser = argparse.ArgumentParser()

#通用参数
parser.add_argument("-d","--dataset_name",default='CAIL',type=str,choices=['CAIL'])  #数据集名称，与README.md和config.py对应

parser.add_argument("-up","--use_preprocessed",action='store_true')  #是否使用预处理后存储在本地的数据（路径由config.py指定）

parser.add_argument("-a","--analyse",action="store_true")  #是否打印对数据集的分析内容

parser.add_argument('-ws','-word_segmentation',default='jieba',nargs='+')  #分词工具，第一个入参是工具名称，后面的入参是其他参数

parser.add_argument("-dp","--do_preprocess",default=None,choices=[None,'use_preprocessed','default'])  #对数据进行预处理工作
#可选方法：负采样，过采样

parser.add_argument("-ps","--preprocess_store",action='store_true')  
#储存数据预处理后的结果到config.py指定路径的文件夹中，方便下次使用。如do_preprocess=None将忽略该参数

parser.add_argument('-we','--word embedding',default='tfidf')  #词嵌入方法。如使用预训练模型将忽略此参数
#可选参数：tfidf skipgram glove fasttext elmo

parser.add_argument("-m","--model",default=None)  #使用的模型。如置None则为不运行模型（仅做数据分析和预处理等）

parser.add_argument('-s','--sub_tasks',default='all')  #需要实现的子任务（有些模型将会忽视此参数）

parser.add_argument('-j','--joint_learning',action='store_true')

parser.add_argument('-dv','--gpu_device',default='cuda:0')  #这个只要是torch.device()可以接受的参数就行了

parser.add_argument('-b','--batch_size',default=128,type=int)

parser.add_argument('-o','--optimizer',default='Adam')

parser.add_argument('-l','--learning_rate',default=0.001,type=float)

#不固定的参数
parser.add_argument('-oa','--other_arguments',nargs='*')
#我还在考虑要不要用字典的文本格式直接传入这个，包括损失函数（交叉熵或focal loss），激活函数，model-specific的超参，ensemble

args = parser.parse_args()
arg_dict=args.__dict__
print(arg_dict)

dataset_name=arg_dict['dataset_name']
isAnalyse=arg_dict['analyse']

import sys,os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import config
from torch_ljp.dataset_utils import cail_analyse

if isAnalyse:
    if dataset_name=='CAIL':
        cail_analyse(data_path=config.cail_original_path,accu_path=config.cail_accu_path,law_path=config.cail_law_path)

