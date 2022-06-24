我不知道怎么解释怎么用了，直接看example.md吧
本文中未列出的情况将报错处理
- CAIL
    - all（默认配置）：去重后，随机抽取70%的数据集作为训练集，10%的数据集作为验证集，20%的数据集作为测试集（数据集比例比较靠近CAIL-small的数据集比例）。
    - big：使用first_stage/train.json作为训练集，first_stage/test.json + restData/rest_data.json作为测试集（CAIL2018原始论文中的数据集配置。类似EPM论文中的CAIL-big数据集，但是样本数上有出入）
    - small：使用exercise_contest/data_train.json作为训练集，exercise_contest/data_valid.json作为验证集，exercise_contest/data_test.json作为测试集（类似EPM论文中的CAIL-small数据集，但是样本数上有出入）
    random_seed：all配置下的随机种子（必须先写all），如不设置，默认为14530529;
    term_split：term of penalty的划分方式，如不设置，默认为'split11'，即将term of penalty划分为11类，类似EPM的实验设置
- ILSI
- ILDC
- ECHR
    - Precedent（参考代码：[valvoda/Precedent](https://github.com/valvoda/Precedent)）