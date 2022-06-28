import json
from tqdm import tqdm

def split11(dataset:list):
    """
    用于CAIL数据集，将term of penalty属性划分为11组，做multi-class one-label分类任务用
    入参是一个由cail_json2str()得到的字符串组成的list
    返回的是一个由dict组成的list，每个元素的键是：fact, charge, article, term, criminals
    """
    return [split11_divide(x) for x in dataset]

def split11_divide(x:str):
    """用于split()中，将一个字符串转换为dict，并划分好11类"""
    d=json.loads(x)
    newd={'fact':d['fact'],'charge':d['meta']['accusation'],'article':d['meta']['relevant_articles'],'criminals':d['meta']['criminals']}
    imprisonment=d['meta']['term_of_imprisonment']
    if imprisonment['death_penalty'] or imprisonment['life_imprisonment']:
        newd['term']='0'
    elif imprisonment['imprisonment']==0:
        newd['term']='1'
    elif imprisonment['imprisonment']<=6:
        newd['term']='2'
    elif imprisonment['imprisonment']<=9:
        newd['term']='3'
    elif imprisonment['imprisonment']<=12:
        newd['term']='4'
    elif imprisonment['imprisonment']<=24:
        newd['term']='5'
    elif imprisonment['imprisonment']<=36:
        newd['term']='6'
    elif imprisonment['imprisonment']<=60:
        newd['term']='7'
    elif imprisonment['imprisonment']<=84:
        newd['term']='8'
    elif imprisonment['imprisonment']<=120:
        newd['term']='9'
    else:
        newd['term']='10'
    return newd

def cail_json2str(d):
    j=json.loads(d)
    #TODO: 感觉这样太蠢了，想办法搞成那种递归根据键值顺序排列的做法试试
    return(json.dumps({"fact":j["fact"],
                        "meta":{"relevant_articles":sorted(j["meta"]["relevant_articles"]),
                                "accusation":sorted(j["meta"]["accusation"]),
                                "punish_of_money":j["meta"]["punish_of_money"],
                                "criminals":sorted(j["meta"]["criminals"]),
                                "term_of_imprisonment":{"death_penalty":j["meta"]["term_of_imprisonment"]["death_penalty"],
                                                        "imprisonment":j["meta"]["term_of_imprisonment"]["imprisonment"],
                                                        "life_imprisonment":j["meta"]["term_of_imprisonment"]["life_imprisonment"]}}},
                    ensure_ascii=False))

def cail2text_cls(data_dict:dict):
    """将经数据集划分后的CAIL数据集的data_dict中的每个值（dict组成的list）的每个元素转换为键为text, charge, article, term的字典"""
    #TODO：软化split11
    newd={}
    for kvpair in data_dict.items():
        newd[kvpair[0]]=[{'fact':x['fact'],'charge':x['charge'],'article':x['article'],'term':x['term']} for x in split11(kvpair[1])]
    return newd

def fasttext_preprocess(task:str,train_file_path:str,test_file_path:str,dataset_dict:dict,word_tokenization:list):
    if word_tokenization[0]=='jieba':
        import jieba
        tk=lambda x:' '.join(jieba.cut(x))
    task_key_map={'law-article-prediction':'article','charge-prediction':'charge','term-of-penalty-prediction':'term'}
    with open(train_file_path,'w') as f:
        for key_name in ['train_set','val_set']:
            #因为fastText不用验证集，所以训练集和验证集全用作训练集
            if key_name in dataset_dict:
                for sample in tqdm(dataset_dict[key_name]):
                    f.write(tk(sample['fact']))
                    for task_label in sample[task_key_map[task]]:
                        f.write(' __label__'+str(task_label))
                    f.write('\n')
    print('成功储存训练数据集在'+train_file_path+'路径')
    with open(test_file_path,'w') as f:
        for sample in tqdm(dataset_dict['test_set']):
            f.write(tk(sample['fact']))
            for task_label in sample[task_key_map[task]]:
                f.write(' __label__'+str(task_label))
            f.write('\n')
    print('成功储存测试数据集在'+test_file_path+'路径')