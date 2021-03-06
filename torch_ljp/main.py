import argparse
parser = argparse.ArgumentParser()

#参数的使用介绍请参考configs文件夹对应参数全称的文件

#通用参数
parser.add_argument("-d","--dataset_name",default=['CAIL','ILSI'],nargs='+')
#第一个参数是数据集名称，与README.md中的数据集名称对应
#后面的参数是使用数据集的不同配置。如不使用-up参数，将根据不同的配置对数据集进行不同的处理。如使用-up参数，将直接忽略其功能
#但这些参数都将出现在命名中
#对不同配置的介绍见configs/dataset_name.md文件

parser.add_argument("-up","--use_preprocessed",default=None)  #使用预处理后的数据，如传入字符串格式的路径，将直接使用
#要求文件夹中

parser.add_argument("-a","--analyse",action="store_true")  #是否打印对数据集的分析内容

parser.add_argument('-ws','--word_segmentation',default='NLTK',nargs='+')  #分词工具，第一个入参是工具名称，后面的入参是其他参数
#英文：NLTK
#中文：jieba

parser.add_argument("-dp","--do_preprocess",default=None,choices=[None,'use_preprocessed','default'])  #对数据进行预处理工作
#可选方法：负采样，过采样

parser.add_argument("-ps","--preprocess_store",action='store_true')  
#储存数据预处理后的结果到config.py指定路径的文件夹中，方便下次使用。如do_preprocess=None将忽略该参数

parser.add_argument('-we','--word embedding',default='tfidf')  #词嵌入方法。如使用预训练模型将忽略此参数
#可选参数：tfidf skipgram glove fasttext elmo

parser.add_argument("-m","--model",default=None)  #使用的模型。如置None则为不运行模型（仅做数据分析和预处理等）

parser.add_argument('--reapper',action='store_true')
#是否需要全局PyTorch保持可复现性

parser.add_argument('-rs','--reappear_seed',default=19390901,type=int)
#可复现性所使用的随机种子，要求是整数

parser.add_argument('-da','--detect_anomaly',action='store_true')
#是否开启PyTorch.autograd的异常检测功能

parser.add_argument('--mode',default='pipeline',choices=['pipeline','train','test'])  #流程模式。全流程（训练+验证+测试）、训练、测试/tuili

parser.add_argument('-s','--sub_tasks',default='multi-task3')  #需要实现的子任务（需要对应数据集和模型）
#multi-task3：用multi-task范式训练3个任务：law article prediction + charge prediction + term of penalty prediction
#law-article-prediction
#charge-prediction
#term-of-penalty-prediction

parser.add_argument('-dv','--gpu_device',default='cuda:0')  #这个只要是torch.device()可以接受的参数就行了

parser.add_argument('-b','--batch_size',default=128,type=int)

parser.add_argument('-o','--optimizer',default='Adam')

parser.add_argument('-l','--learning_rate',default=0.001,type=float)

#不固定的参数
parser.add_argument('-oa','--other_arguments',nargs='*')
#包括损失函数（交叉熵或focal loss），激活函数，model-specific的超参，ensemble

args = parser.parse_args()
arg_dict=args.__dict__
configuration_log=str(arg_dict)  #用str格式保存
print(arg_dict)

import sys,os,random,torch
from tqdm import tqdm
from datetime import datetime
import numpy as np
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import config
from torch_ljp.dataset_utils.split import cail_split

dataset_name=arg_dict['dataset_name'][0]
isAnalyse=arg_dict['analyse']
model_name=arg_dict['model']
other_arguments=arg_dict['other_arguments']
sub_tasks=arg_dict['sub_tasks']
mode=arg_dict['mode']
word_segmentation=arg_dict['word_segmentation']
SEED=arg_dict['reappear_seed']

print('配置异常检测和复现环境...')
if arg_dict['detect_anomaly']:
    torch.autograd.set_detect_anomaly(True)
if arg_dict['reapper']:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

#划分数据集：目前的做法还是直接把所有数据集对象加载到内存中，以后再研究有没有什么更好的方法
if arg_dict['use_preprocessed']:
    pass
else:
    print('数据划分ing...')
    if dataset_name=='CAIL':
        dataset_dict=cail_split(data_path=config.cail_original_path,data_config=arg_dict['dataset_name'][1:])
#这一步将数据集划分为训练集-(验证集)-测试集（字典）

if isAnalyse:
    print('\n数据分析ing...')
    if dataset_name=='CAIL':
        from torch_ljp.dataset_utils import cail_analyse
        cail_analyse(data_dict=dataset_dict,accu_path=config.cail_accu_path,law_path=config.cail_law_path,
                    data_config=arg_dict['dataset_name'][1:])

#数据转换→运行模型→计算指标或输出其他结果
if model_name:
    print('模型处理ing...')
    
    if model_name in ['fastText']:  #general-domain文本分类模型
        if dataset_name=='CAIL':
            from torch_ljp.dataset_utils.preprocess import cail2text_cls
            dataset_dict=cail2text_cls(dataset_dict)
        #在这一步将数据集字典的每个值转换为以JSON为元素的列表
        
        if model_name=='fastText':
            import fasttext
            from torch_ljp.dataset_utils.preprocess import fasttext_preprocess

            if os.path.isdir(other_arguments[0]):
                #以文件夹为入参
                print('预处理fastText数据ing...')
                train_file_path=os.path.join(other_arguments[0],
                        dataset_name+'_'.join(arg_dict['dataset_name'][1:])+'_'+sub_tasks+'_train'+\
                                                                                str(datetime.now()).replace('.','_').replace(' ','_')+'.txt')
                test_file_path=os.path.join(other_arguments[0],
                        dataset_name+'_'.join(arg_dict['dataset_name'][1:])+'_'+sub_tasks+'_test'+\
                                                                                str(datetime.now()).replace('.','_').replace(' ','_')+'.txt')
                fasttext_preprocess(sub_tasks,train_file_path,test_file_path,dataset_dict,word_tokenization=word_segmentation)
            else:
                #以2个文件为入参
                train_file_path=other_arguments[0]
                test_file_path=other_arguments[1]

                if len(other_arguments)>2 and other_arguments[2]=='recal':
                    print('预处理fastText数据ing...')
                    fasttext_preprocess(sub_tasks,train_file_path,test_file_path,dataset_dict,word_tokenization=word_segmentation)

            if not mode=='test':
                #训练
                fasttext_model=fasttext.train_supervised(train_file_path)
                #TODO: 储存模型
            if not mode=='train':
                #TODO: 测试
                print(fasttext_model.test(test_file_path))
                

            

                