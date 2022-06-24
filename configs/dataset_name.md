每个数据集下第一个缩进是第一个入参，第二个是第二个入参，以此类推。
本文中未列出的情况将报错处理
- CAIL
    - all（默认配置）：去重后，随机抽取70%的数据集作为训练集，10%的数据集作为验证集，20%的数据集作为测试集（数据集比例比较靠近CAIL-small的数据集比例）。
        随机种子：如不设置，默认为14530529
    - big：使用first_stage/train.json作为训练集，first_stage/test.json + restData/rest_data.json作为测试集（CAIL2018原始论文中的数据集配置。类似EPM论文中的CAIL-big数据集，但是样本数上有出入）
    - small：使用exercise_contest/data_train.json作为训练集，exercise_contest/data_valid.json作为验证集，exercise_contest/data_test.json作为测试集（类似EPM论文中的CAIL-small数据集，但是样本数上有出入）
    - bigLADAN
    - smallLADAN
- ILDC
- ECHR
    - Precedent（参考代码：[valvoda/Precedent](https://github.com/valvoda/Precedent)）