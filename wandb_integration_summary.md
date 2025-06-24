# Wandb集成总结

## 概述

已成功将Wandb（Weights & Biases）集成到GraphCodeBERT克隆检测项目中，实现了训练、评估和测试过程中指标的自动记录和跟踪。

## 修改的文件

### 1. `run.py` - 主要修改

#### 新增功能：
- **Wandb初始化**: 在训练开始时初始化wandb，记录超参数配置
- **训练指标记录**: 记录训练损失、学习率、轮次等
- **评估指标记录**: 记录验证集上的召回率、精确率、F1分数
- **最佳模型跟踪**: 记录训练过程中的最佳模型性能
- **测试指标记录**: 记录测试集上的最终性能指标
- **自动关闭**: 训练/评估/测试完成后自动关闭wandb会话

#### 新增命令行参数：
- `--use_wandb`: 启用wandb日志记录
- `--wandb_project`: 指定wandb项目名称
- `--wandb_run_name`: 指定wandb运行名称

### 2. 新增文件

#### `wandb_integration_guide.md`
- 详细的使用指南
- 功能说明
- 示例命令
- 故障排除

#### `run_with_wandb.sh`
- 完整的训练、评估、测试脚本示例
- 展示如何使用wandb参数

#### `test_wandb_integration.py`
- 集成测试脚本
- 验证wandb功能是否正常工作

#### `wandb_integration_summary.md`
- 本文档，总结所有修改

## 记录的指标

### 训练过程指标
```
train/loss              # 训练损失
train/learning_rate     # 学习率
train/epoch            # 当前轮次
train/global_step      # 全局步数
```

### 评估指标
```
eval/recall            # 召回率
eval/precision         # 精确率
eval/f1               # F1分数
eval/threshold        # 分类阈值
```

### 最佳模型指标
```
best/f1               # 最佳F1分数
best/recall           # 最佳召回率
best/precision        # 最佳精确率
best/step             # 达到最佳性能的步数
```

### 最终评估指标
```
final_eval/recall     # 最终评估召回率
final_eval/precision  # 最终评估精确率
final_eval/f1        # 最终评估F1分数
final_eval/loss      # 最终评估损失
```

### 测试指标
```
test/recall          # 测试召回率
test/precision       # 测试精确率
test/f1             # 测试F1分数
test/loss           # 测试损失
test/threshold      # 测试阈值
```

## 使用方法

### 1. 基本使用

```bash
# 训练时启用wandb
python run.py --do_train --use_wandb --wandb_project "my-project" [其他参数...]

# 评估时启用wandb
python run.py --do_eval --use_wandb --wandb_project "my-project" [其他参数...]

# 测试时启用wandb
python run.py --do_test --use_wandb --wandb_project "my-project" [其他参数...]
```

### 2. 使用提供的脚本

```bash
# 设置wandb API key
export WANDB_API_KEY="your_api_key"

# 运行完整流程
./run_with_wandb.sh
```

### 3. 测试集成

```bash
python test_wandb_integration.py
```

## 优势

1. **自动记录**: 无需手动记录指标，自动上传到wandb
2. **可视化**: 在wandb dashboard中查看训练曲线和指标
3. **实验管理**: 轻松比较不同实验的结果
4. **超参数跟踪**: 自动记录所有训练超参数
5. **协作**: 团队成员可以共享和查看实验结果

## 兼容性

- **向后兼容**: 不使用`--use_wandb`参数时，代码行为与原来完全一致
- **可选功能**: wandb是可选功能，不影响核心训练逻辑
- **错误处理**: 即使wandb出现问题，训练过程也会继续

## 注意事项

1. **网络要求**: 需要网络连接来上传数据到wandb
2. **API Key**: 需要设置有效的wandb API key
3. **存储**: wandb会占用一定的网络带宽和存储空间

## 验证结果

运行`test_wandb_integration.py`的结果：
```
✓ All tests passed! Wandb integration is ready to use.
```

所有测试通过，确认集成成功且功能正常。

## 下一步

1. 设置wandb账户和API key
2. 运行训练并在wandb dashboard中查看结果
3. 根据需要调整记录的指标或添加新的指标
4. 利用wandb的高级功能如超参数搜索、模型版本管理等
