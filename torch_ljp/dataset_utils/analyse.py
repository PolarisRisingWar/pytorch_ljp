#本项目分析数据集的方法参考了以下论文中对数据集进行描述的部分：
# 1. CAIL2018: A Large-Scale Legal Dataset for Judgment Prediction

#TODO：打印文本的句子数、词数、char数、token数

import random,os,json,re
from tqdm import tqdm

from torch_ljp.general_utils.an_to_cn import an2cn

dataset_utils_path=os.path.dirname(os.path.realpath(__file__))
cn_crimnal_law_path=os.path.join(dataset_utils_path,'other_data','cn_criminal_law.txt')

def cail_analyse(data_dict:dict,accu_path:str="",law_path:str="",data_config:list=[]):
    """
    对CAIL数据集进行分析
    """
    #打印已知基本信息
    print('CAIL数据集：中文刑事案件数据集，源自中国裁判文书网http://wenshu.court.gov.cn/ \n虽然原文声称被告只有一个，但其实存在多被告场景')
    print()

    accu=[x.strip() for x in open(accu_path).readlines()]
    accu_counts=[0 for _ in range(len(accu))]  #每个accu标签对应的样本数：去重+multi-label
    law=[x.strip() for x in open(law_path).readlines()]
    law_counts=[0 for _ in range(len(law))]
    
    #检查训练集/验证集/测试集之间的交集（数据泄露情况）
    #注意：非all情况的交集数只能作为参考
    try:
        intersection=set.intersection(set(data_dict['train_set']),set(data_dict['val_set']),set(data_dict['test_set']))
        print('训练集样本数：'+str(len(data_dict['train_set']))+'\n验证集样本数：'+str(len(data_dict['val_set'])))
        print('测试集样本数：'+str(len(data_dict['test_set'])))
        print('训练集、验证集、测试集之间的交集样本数：'+str(len(intersection)))
        deduplicate_list=data_dict['train_set']+data_dict['val_set']+data_dict['test_set']
    except KeyError:  #没有验证集的情况，即数据集划分为big时
        intersection=set.intersection(set(data_dict['train_set']),set(data_dict['test_set']))
        print('训练集样本数：'+str(len(data_dict['train_set']))+'\n测试集样本数：'+str(len(data_dict['test_set'])))
        print('训练集、测试集之间的交集样本数：'+str(len(intersection)))
        deduplicate_list=data_dict['train_set']+data_dict['test_set']
    #small：25

    #随机选一条数据，打印样本示例
    #事实描述文本和判决结果
    chosen_json=json.loads(random.choice(deduplicate_list))
    print('样本原文：'+str(chosen_json)+'\n')
    print('样本内容解释：\n\n事实描述文本：'+chosen_json['fact']+'\n被告：'+str(chosen_json['meta']['criminals']))
    print('罚款：'+str(chosen_json['meta']['punish_of_money'])+'\n罪名：'+str(chosen_json['meta']['accusation']))
    articles=chosen_json['meta']['relevant_articles']
    cn_crimnal_law=open(cn_crimnal_law_path).read()
    for an_article in articles:
        cn_article=an2cn(an_article)
        article_text_begin_index=cn_crimnal_law.find('　　第'+cn_article+'条　【')
        article_text_end_index1=cn_crimnal_law.find('　　第'+an2cn(int(an_article)+1)+'条　【')  #一种可能性是下一条
        if article_text_end_index1<0:
            article_text_end_index1=len(cn_crimnal_law)
        article_text_end_index2=0  #另一种可能性是编/章/节
        for obj in re.finditer('第.[编章节]',cn_crimnal_law[article_text_begin_index:]):
            article_text_end_index2=obj.span()[0]+article_text_begin_index
            break
        if article_text_end_index2<1:
            article_text_end_index2=len(cn_crimnal_law)
        article_text_end_index=min(article_text_end_index1,article_text_end_index2)
        print('对应法条编号：'+str(an_article)+'  刑法原文：\n'+cn_crimnal_law[article_text_begin_index:article_text_end_index].strip()+'\n')
    print('是否死刑：'+('是' if chosen_json['meta']['term_of_imprisonment']['death_penalty'] else '否'))
    print('是否无期：'+('是' if chosen_json['meta']['term_of_imprisonment']['life_imprisonment'] else '否'))
    print('有期徒刑刑期（单位：月）：'+str(chosen_json['meta']['term_of_imprisonment']['imprisonment'])+'\n')

    #检查每种accusation/article对应的样本数，每个样本的平均被告数、article数、accusation数
    #TODO: 根据这个列表来画个分布图，以体现出数据集的分布不平衡这一特性来
    criminal_sum=0
    article_sum=0
    accusation_sum=0

    #对term of penalty的处理
    if len(data_config)==0 or ('term_split' not in data_config) or (data_config[data_config.index('term_split')+1]=='split11'):
        #TODO: 处理非split11的情况
        print('将term of penalty处理为11类的形式')
        from torch_ljp.dataset_utils.preprocess import split11
        split11_list=split11(deduplicate_list)
        split11_counts=[0 for _ in range(11)]
    else:
        split11_list=[json.loads(x) for x in deduplicate_list]
        #其实还没考虑这种情况

    for factor in tqdm(split11_list):
        article_list=factor['article']
        article_sum+=len(article_list)
        for a in article_list:
            law_index=law.index(str(a))  #这个我发现就有的是str，有的是int，比较的狗
            law_counts[law_index]+=1
        accusation=factor['charge']
        accusation_sum+=len(accusation)
        for a in accusation:
            accu_index=accu.index(re.sub('[\[\]]','',a))  #边缘例子：'[生产、销售]伪劣产品'
            accu_counts[accu_index]+=1
        criminals=factor["criminals"]
        criminal_sum+=len(criminals)
        split11_counts[int(factor['term'])]+=1  #TODO: 考虑不用split11的情况
        
    
    print(law_counts)
    print(accu_counts)
    print(criminal_sum/len(deduplicate_list))
    print(accusation_sum/len(deduplicate_list))
    print(criminal_sum/len(deduplicate_list))  #这三个基本上都是1左右
    print(split11_counts)



def ilsi_analyse(data_path:str,up:bool=False,data_config:list=[]):
    return