打印数据分析内容：
- CAIL all 划分数据集用默认随机种子，默认配置（将term of penalty划分为11类）：`python torch_ljp/main.py -a` 或 `python torch_ljp/main.py -d CAIL -a` 或 `python torch_ljp/main.py -d CAIL all -a` 输出示例：op_examples/analyse/cail-all.out
- CAIL all 划分数据集用随机种子42，默认配置（将term of penalty划分为11类）：`python torch_ljp/main.py -d CAIL all random_seed 42 -a` 输出示例：op_examples/analyse/cail-all.out
- CAIL small 默认配置（将term of penalty划分为11类）：`python torch_ljp/main.py -d CAIL small -a` 输出示例：op_examples/analyse/cail-small.out
- CAIL big 默认配置（将term of penalty划分为11类）：`python torch_ljp/main.py -d CAIL big -a` 输出示例：op_examples/analyse/cail-big.out

fastText分类：
- CAIL small 默认配置（将term of penalty划分为11类，测试阶段输出1个标签），做law article prediction任务，使用jieba包分词，全流程，中间数据储存在path路径（文件夹）上：`python torch_ljp/main.py -d CAIL small -ws jieba -m fastText -s law-article-prediction -oa path`
- CAIL small 默认配置（将term of penalty划分为11类，测试阶段输出1个标签），做law article prediction任务，使用jieba包分词，全流程，使用训练集数据文件path1和测试集数据文件path2：`python torch_ljp/main.py -d CAIL small -ws jieba -m fastText -s law-article-prediction -oa path1 path2`
- CAIL small 默认配置（将term of penalty划分为11类，测试阶段输出1个标签），做law article prediction任务，使用jieba包分词，全流程，中间数据存储在path1和path2：`python torch_ljp/main.py -d CAIL small -ws jieba -m fastText -s law-article-prediction -oa path1 path2 recal`