# SS71 实验目录

## 实验结果
output/BP、output/MF、output/CC 分别对应三个GO任务。
每个任务下：
- onehot：SS8 One-hot编码实验。
- embed：SS8 Embedding编码实验。
- comparison.csv：该任务两种方案的对比结果。

每个方案目录保留：
best_model.pth、results.csv、loss.png。

## 训练与数据
训练入口：train.py。
模型、数据加载和配置仍由根目录原有脚本提供。
继续使用原来的PyTorch运行环境和训练命令。
原来的 *_full_20epochs 输出参数会自动归一化为新的短路径。
重新训练同一任务和方案时，请另设新的实验输出根目录，
避免覆盖已归档模型。

data 中的FASTA、SS8、标签和注释未修改。

## SPOT-1D-LM
保留了工具代码、预训练模型及现有输入/预测材料。
其中inputs体积很大；清理前需要确认具体缓存类型和复用需求。
训练阶段读取data中的最终SS8。

## 整理记录
records/layout.json：迁移和文件校验记录。
records/layout_backup：原train.py及原README备份。
logs：从根目录迁移的.log文件。
