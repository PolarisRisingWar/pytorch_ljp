import json

def split11(dataset:list):
    """
    用于CAIL数据集将term of penalty属性划分为11组，做multi-class one-label分类任务用
    入参是一个由cail_json2str()得到的字符串组成的list
    返回的是一个由dict组成的list，每个元素的键是：fact, 
    """
    return dataset

def split11_divide(x:str):
    """用于split()中，将一个字符串转换为dict，并划分好11类"""
    d=json.loads(x)
    newd={'fact':d['fact'],'charge':d['meta']['accusation'],'article':d['meta']['relevant_articles']}
    