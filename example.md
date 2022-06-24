打印数据分析内容：
- CAIL all 划分数据集用默认随机种子，默认配置（将term of penalty划分为11类）：`python torch_ljp/main.py -a` 输出示例：op_examples/analyse/cail-all.out
- CAIL all 划分数据集用随机种子42，默认配置（将term of penalty划分为11类）：`python torch_ljp/main.py -d CAIL all random_seed 42 -a` 输出示例：op_examples/analyse/cail-all.out
- CAIL small，默认配置（将term of penalty划分为11类）：`python torch_ljp/main.py -d CAIL small -a` 输出示例：op_examples/analyse/cail-small.out
- CAIL big，默认配置（将term of penalty划分为11类）：`python torch_ljp/main.py -d CAIL big -a` 输出示例：op_examples/analyse/cail-big.out

fastText分类：
- CAIL small 默认配置（测试阶段输出1个标签），使用jieba包分词，全流程，中间数据储存在path路径上：`python torch_ljp/main.py -d CAIL small -ws jieba 