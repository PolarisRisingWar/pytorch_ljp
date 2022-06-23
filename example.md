打印数据分析内容：
- CAIL all 划分数据集用默认随机种子：`python torch_ljp/main.py -a` 输出示例：op_examples/analyse/cail-all.out
- CAIL all 划分数据集用随机种子42：`python torch_ljp/main.py -d CAIL all 42 -a` 输出示例：op_examples/analyse/cail-all.out
- CAIL small：`python torch_ljp/main.py -d CAIL small -a` 输出示例：op_examples/analyse/cail-small.out
- CAIL big：`python torch_ljp/main.py -d CAIL big -a` 输出示例：op_examples/analyse/cail-big.out