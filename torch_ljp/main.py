import argparse
parser = argparse.ArgumentParser()

#参数的使用介绍请参考configs文件夹对应参数全称的文件

#通用参数
parser.add_argument("-d","--dataset_name",default=['CAIL'],nargs='+')
#第一个参数是数据集名称，与README.md中的数据集名称对应
#后面的参数是使用数据集的不同配置。如不使用-up参数，将根据不同的配置对数据集进行不同的处理。如使用-up参数，将直接忽略其功能
#但这些参数都将出现在命名中
#对不同配置的介绍见configs/dataset_name.md文件

parser.add_argument("-up","--use_preprocessed",default=None)  #使用预处理后的数据，如传入字符串格式的路径，将直接使用
#要求文件夹中

parser.add_argument("-a","--analyse",action="store_true")  #是否打印对数据集的分析内容

parser.add_argument('-ws','-word_segmentation',default='NLTK',nargs='+')  #分词工具，第一个入参是工具名称，后面的入参是其他参数
#英文：NLTK
#中文：jieba

parser.add_argument("-dp","--do_preprocess",default=None,choices=[None,'use_preprocessed','default'])  #对数据进行预处理工作
#可选方法：负采样，过采样

parser.add_argument("-ps","--preprocess_store",action='store_true')  
#储存数据预处理后的结果到config.py指定路径的文件夹中，方便下次使用。如do_preprocess=None将忽略该参数

parser.add_argument('-we','--word embedding',default='tfidf')  #词嵌入方法。如使用预训练模型将忽略此参数
#可选参数：tfidf skipgram glove fasttext elmo

parser.add_argument("-m","--model",default=None)  #使用的模型。如置None则为不运行模型（仅做数据分析和预处理等）

parser.add_argument('--mode',default='pipeline',choices=['pipeline','train','test'])  #流程模式。全流程（训练+验证+测试）、训练、测试/tuili

parser.add_argument('-s','--sub_tasks',default='multi-task3')  #需要实现的子任务（需要对应数据集和模型）
#multi-task3：law article prediction + charge prediction + term of penalty prediction
#law-article-prediction
#chrage-prediction
#term-of-penalty-prediction

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
configuration_log=str(arg_dict)  #用str格式保存
print(arg_dict)

import sys,os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import config
from torch_ljp.dataset_utils.split import cail_split

dataset_name=arg_dict['dataset_name'][0]
isAnalyse=arg_dict['analyse']

#划分数据集：目前的做法还是直接把所有数据集对象加载到内存中，以后再研究有没有什么更好的方法
if arg_dict['use_preprocessed']:
    pass  #TODO: 做这个
else:
    print('数据划分阶段：')
    if dataset_name=='CAIL':
        dataset_dict=cail_split(data_path=config.cail_original_path,data_config=arg_dict['dataset_name'][1:])

if isAnalyse:
    print('\n数据分析阶段：')
    if dataset_name=='CAIL':
        from torch_ljp.dataset_utils import cail_analyse
        cail_analyse(data_dict=dataset_dict,accu_path=config.cail_accu_path,law_path=config.cail_law_path,
                    data_config=arg_dict['dataset_name'][1:])

