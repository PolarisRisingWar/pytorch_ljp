#本项目分析数据集的方法参考了以下论文中对数据集进行描述的部分：
# 1. CAIL2018: A Large-Scale Legal Dataset for Judgment Prediction

import random,os,json,re
from tqdm import tqdm

from torch_ljp.general_utils.an_to_cn import an2cn

dataset_utils_path=os.path.dirname(os.path.realpath(__file__))
cn_crimnal_law_path=os.path.join(dataset_utils_path,'other_data','cn_criminal_law.txt')

def cail_analyse(data_path:str,accu_path:str="",law_path:str=""):
    """
    对CAIL数据集进行分析
    """
    #打印已知基本信息
    print('CAIL数据集：中文刑事案件数据集，源自中国裁判文书网http://wenshu.court.gov.cn/ \n虽然原文声称被告只有一个，但其实存在多被告场景')
    print()

    train_small_path=os.path.join(data_path,'exercise_contest','data_train.json')
    valid_small_path=os.path.join(data_path,'exercise_contest','data_valid.json')
    test_small_path=os.path.join(data_path,'exercise_contest','data_test.json')
    train_big_path=os.path.join(data_path,'first_stage','train.json')
    test_big_path=os.path.join(data_path,'first_stage','test.json')
    final_test_path=os.path.join(data_path,'final_test.json')
    rest_path=os.path.join(data_path,'restData','rest_data.json')
    subpaths=[train_small_path,valid_small_path,test_small_path,train_big_path,test_big_path,final_test_path,rest_path]

    accu=[x.strip() for x in open(accu_path).readlines()]
    accu_counts=[0 for _ in range(len(accu))]  #每个accu标签对应的样本数：去重+multi-label
    law=[x.strip() for x in open(law_path).readlines()]
    law_counts=[0 for _ in range(len(law))]

    #随机选一条数据，打印样本示例
    #事实描述文本和判决结果
    chosen_json=json.loads(random.choice(open(random.choice(subpaths)).readlines()))
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

    #官方README文件中说一共有268万条数据，但我数出来其实要更多一些。为什么会这样，小编也很好奇
    #分别统计去重和不去重条件下的样本总数
    duplicate_no=0
    deduplicate=set()
    ratio=set()
    for subpath in subpaths:
        thelist=open(subpath).readlines()
        print(subpath+'文件中包含的数据数为：'+str(len(thelist)))
        duplicate_no+=len(thelist)
        deduplicate.update([cail_mediate_func(i) for i in thelist])
    print('不去重样本总数：'+str(duplicate_no))  #2916228
    print('去重样本总数：'+str(len(deduplicate)))  #2784403

    #检查每种accusation/article对应的样本数，每个样本的平均被告数、article数、accusation数
    #TODO: 根据这个列表来画个分布图，以体现出数据集的分布不平衡这一特性来
    criminal_sum=0
    article_sum=0
    accusation_sum=0
    for factor in tqdm(deduplicate):
        article_list=eval(factor[factor.find('article:')+8:factor.find('accu:')])
        article_sum+=len(article_list)
        for a in article_list:
            law_index=law.index(str(a))  #这个我发现就有的是str，有的是int，比较的狗
            law_counts[law_index]+=1
        accusation=eval(factor[factor.find('accu:')+5:factor.find('criminal:')])
        accusation_sum+=len(accusation)
        for a in accusation:
            accu_index=accu.index(re.sub('[\[\]]','',a))  #边缘例子：'[生产、销售]伪劣产品'
            accu_counts[accu_index]+=1
        criminals=eval(factor[factor.find('criminal:')+9:])
        criminal_sum+=len(criminals)
        
    
    print(law_counts)
    print(accu_counts)
    print(criminal_sum/len(deduplicate))
    print(accusation_sum/len(deduplicate))
    print(criminal_sum/len(deduplicate))  #这三个基本上都是1左右



    


def cail_mediate_func(d):
    j=json.loads(d)
    m=j['meta']
    ti=m['term_of_imprisonment']
    if len(m['criminals'])>1:
        pass  #反正有
    
    return(j['fact']+str(m['punish_of_money'])+str(ti['death_penalty'])+str(ti['life_imprisonment'])+str(ti['imprisonment'])+\
            'article:'+str(sorted(m['relevant_articles']))+'accu:'+str(sorted(m['accusation']))+'criminal:'+str(sorted(m['criminals'])))