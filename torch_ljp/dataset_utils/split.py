import os,random,json

def cail_split(data_path:str,data_config:list=[]):
    """
    划分CAIL数据集，返回dict文件，每个元素是由字符串（可以转换为JSON格式）组成的list，没有经过任何除去重和划分之外的预处理过程
    """
    train_small_path=os.path.join(data_path,'exercise_contest','data_train.json')
    valid_small_path=os.path.join(data_path,'exercise_contest','data_valid.json')
    test_small_path=os.path.join(data_path,'exercise_contest','data_test.json')
    train_big_path=os.path.join(data_path,'first_stage','train.json')
    test_big_path=os.path.join(data_path,'first_stage','test.json')
    final_test_path=os.path.join(data_path,'final_test.json')
    rest_path=os.path.join(data_path,'restData','rest_data.json')

    #划分数据集
    if len(data_config)==0 or data_config[0]=='all':
        filepaths=[train_small_path,valid_small_path,test_small_path,train_big_path,test_big_path,final_test_path,rest_path]

        print('实验中用到的数据及其对应的样本数：')

        #TODO: 以下这种写法会导致样本顺序乱掉。暂时没有保持顺序的需求所以没管这事

        #分别统计各个数据文件的样本数，
        duplicate_no=0
        deduplicate=set()
        for filepath in filepaths:
            thelist=open(filepath).readlines()
            print(filepath+'文件，样本数：'+str(len(thelist)))
            duplicate_no+=len(thelist)
            deduplicate.update([cail_json2str(x) for x in thelist])
        print('不去重样本总数：'+str(duplicate_no))  #2916228
        print('去重样本总数：'+str(len(deduplicate)))  #2784403

        if len(data_config)>1 and 'random_seed' in eval(data_config[1]):
            random.seed(data_config[1]['random_seed'])
        else:
            random.seed(14530529)
        
        deduplicate_list=list(deduplicate)
        random.shuffle(deduplicate_list)
        train_set=deduplicate_list[:int(0.7*len(deduplicate))]
        val_set=deduplicate_list[int(0.7*len(deduplicate)):int(0.8*len(deduplicate))]
        test_set=deduplicate_list[int(0.8*len(deduplicate)):]
        return {'train_set':train_set,'val_set':val_set,'test_set':test_set}
    elif data_config[0]=='big':
        train_set=open(train_big_path).readlines()
        test_set=open(test_big_path).readlines()+open(rest_path).readlines()
        return {'train_set':train_set,'test_set':test_set}
    elif data_config[0]=='small':
        train_set=open(train_small_path).readlines()
        val_set=open(valid_small_path).readlines()
        test_set=open(test_small_path).readlines()
        return {'train_set':train_set,'val_set':val_set,'test_set':test_set}

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
