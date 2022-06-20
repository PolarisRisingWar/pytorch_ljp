本项目旨在使用原生PyTorch统一实现法律判决预测LJP（legal judgment prediction）任务的当前各重要模型，包括对多种语言下多种数据的预处理、多种子任务下的实现。  
直接通过命令行即可调用torch_ljp/main.py文件，传入参数并得到对应的结果，需要预先在torch_ljp文件夹下创建config.py文件（由于真实文件涉及个人隐私，因此没有上传，但是我上传了一个fakeconfig.py文件，把里面需要填的参数填上就行）。  
具体的使用命令可参考example.txt。  

以下分别介绍本项目中已经可实现分析和处理的数据，模型，及二者相对应的任务中，我跑出来的实验结果和原论文或其他引用论文中跑出来的结果的对比（有海量没整好的内容，等我慢慢补吧）：
（如果您希望我添加什么数据或模型，可以直接给我提issue！）
# 1. 数据
中文：
- [x] CAIL（来源：[CAIL2018: A Large-Scale Legal Dataset for Judgment Prediction](https://arxiv.org/abs/1807.02478)，下载地址：<https://cail.oss-cn-qingdao.aliyuncs.com/CAIL2018_ALL_DATA.zip>）
- [ ] LJP-E（还没有完全公开。来源：[Legal Judgment Prediction via Event Extraction with Constraints](https://aclanthology.org/2022.acl-long.48/)）

英文（美国）：
- [ ] ILLDM（作者在论文里说要公开的，但是至今没有公开。来源：[Interpretable Low-Resource Legal Decision Making](https://arxiv.org/abs/2201.01164)）

英文（印度）：
- [ ] ILSI（来源：[LeSICiN: A Heterogeneous Graph-Based Approach for Automatic Legal Statute Identification from Indian Legal Documents](https://arxiv.org/abs/2112.14731)，下载地址：[Dataset and additional files/softwares required for the paper "LeSICiN: A Heterogeneous Graph-based Approach for Automatic Legal Statute Identification from Indian Legal Documents" | Zenodo](https://zenodo.org/record/6053791#.YrAtHnZByUl)（除best_model.pt和ils2v.bin外都是数据相关的文件）

法语（比利时）：
- [ ] BSARD（来源：[A Statutory Article Retrieval Dataset in French](https://arxiv.org/abs/2108.11792)）

# 2. 模型
- [ ] LibSVM（来源：[CAIL2018/baseline at master · thunlp/CAIL2018](https://github.com/thunlp/CAIL2018/tree/master/baseline)）
- [ ] LeSICiN（来源：[LeSICiN: A Heterogeneous Graph-Based Approach for Automatic Legal Statute Identification from Indian Legal Documents](https://arxiv.org/abs/2112.14731)）
- [ ] ILLDM（只能用在特殊数据里，但是原始数据还没有公开。来源：[Interpretable Low-Resource Legal Decision Making](https://arxiv.org/abs/2201.01164)）
- [ ] EPM（官方代码还没有完全公开，我发邮件问了作者他说他以后要全部公开的，所以我想等他们全部公开了再写。来源：[Legal Judgment Prediction via Event Extraction with Constraints](https://aclanthology.org/2022.acl-long.48/)

# 3. 实验结果

其他信息：
1. torch_ljp/dataset_utils/other_data文件夹内放的是一些比较小，而且不太好解释怎么制作的文件，所以直接跟着GitHub项目一起上传了。
    1. cn_criminal_law.txt：2021版中华人民共和国刑法。复制自[中华人民共和国刑法（2022年最新版） - 中国刑事辩护网](http://www.chnlawyer.net/law/subs/xingfa.html)中下载的Word文件，并删除了其中语涉“中国刑事辩护网提供……”的字样。