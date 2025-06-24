# Wandb Integration Guide for GraphCodeBERT Clone Detection

本指南介绍如何在GraphCodeBERT克隆检测项目中使用Wandb来记录和跟踪训练、评估和测试过程中的指标。

## 功能概述

已集成的Wandb功能包括：

### 1. 训练过程指标记录
- **训练损失** (`train/loss`): 每个训练步骤的平均损失
- **学习率** (`train/learning_rate`): 当前学习率
- **训练轮次** (`train/epoch`): 当前训练轮次
- **全局步数** (`train/global_step`): 全局训练步数

### 2. 评估过程指标记录
- **评估召回率** (`eval/recall`): 验证集上的召回率
- **评估精确率** (`eval/precision`): 验证集上的精确率
- **评估F1分数** (`eval/f1`): 验证集上的F1分数
- **评估阈值** (`eval/threshold`): 使用的分类阈值

### 3. 最佳模型指标记录
- **最佳F1分数** (`best/f1`): 训练过程中达到的最佳F1分数
- **最佳召回率** (`best/recall`): 最佳模型的召回率
- **最佳精确率** (`best/precision`): 最佳模型的精确率
- **最佳步数** (`best/step`): 达到最佳性能的训练步数

### 4. 最终评估指标记录
- **最终评估指标** (`final_eval/*`): 在验证集上的最终评估结果
- **测试指标** (`test/*`): 在测试集上的最终测试结果

## 使用方法

### 1. 安装Wandb

```bash
pip install wandb
```

### 2. 设置Wandb API Key

```bash
# 方法1: 使用wandb login命令
wandb login

# 方法2: 设置环境变量
export WANDB_API_KEY="your_wandb_api_key_here"
```

### 3. 训练时启用Wandb

在运行训练命令时添加以下参数：

```bash
python run.py \
    --do_train \
    --use_wandb \
    --wandb_project "your-project-name" \
    --wandb_run_name "your-run-name" \
    # ... 其他训练参数
```

### 4. 评估时启用Wandb

```bash
python run.py \
    --do_eval \
    --use_wandb \
    --wandb_project "your-project-name" \
    --wandb_run_name "eval-run" \
    # ... 其他评估参数
```

### 5. 测试时启用Wandb

```bash
python run.py \
    --do_test \
    --use_wandb \
    --wandb_project "your-project-name" \
    --wandb_run_name "test-run" \
    # ... 其他测试参数
```

## 新增的命令行参数

- `--use_wandb`: 启用Wandb日志记录（布尔标志）
- `--wandb_project`: Wandb项目名称（默认: "graphcodebert-clonedetection"）
- `--wandb_run_name`: Wandb运行名称（可选，如果不指定会自动生成）

## 示例使用

### 完整训练流程示例

```bash
# 训练
python run.py \
    --output_dir=./saved_models \
    --config_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --do_train \
    --train_data_file=dataset/train.txt \
    --eval_data_file=dataset/valid.txt \
    --epochs 2 \
    --code_length 512 \
    --data_flow_length 128 \
    --train_batch_size 6 \
    --eval_batch_size 8 \
    --learning_rate 2e-5 \
    --evaluate_during_training \
    --use_wandb \
    --wandb_project "graphcodebert-clonedetection" \
    --wandb_run_name "train-experiment-1"

# 测试
python run.py \
    --output_dir=./saved_models \
    --config_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --do_test \
    --test_data_file=dataset/test.txt \
    --code_length 512 \
    --data_flow_length 128 \
    --eval_batch_size 8 \
    --use_wandb \
    --wandb_project "graphcodebert-clonedetection" \
    --wandb_run_name "test-experiment-1"
```

## Wandb Dashboard中的指标

在Wandb Dashboard中，你可以看到以下图表和指标：

1. **训练损失曲线**: 显示训练过程中损失的变化
2. **学习率调度**: 显示学习率随时间的变化
3. **评估指标**: 显示验证集上的性能指标
4. **最佳模型跟踪**: 跟踪训练过程中的最佳模型性能
5. **超参数记录**: 自动记录所有训练超参数

## 注意事项

1. **网络连接**: 确保有稳定的网络连接以上传日志到Wandb
2. **API Key**: 确保正确设置了Wandb API Key
3. **项目权限**: 确保有权限访问指定的Wandb项目
4. **存储空间**: Wandb会存储日志数据，注意账户的存储限制

## 故障排除

### 常见问题

1. **Wandb未初始化错误**
   - 确保设置了`--use_wandb`参数
   - 检查API Key是否正确设置

2. **网络连接问题**
   - 检查网络连接
   - 如果在代理环境下，确保代理设置正确

3. **权限问题**
   - 确保有权限访问指定的Wandb项目
   - 检查API Key是否有效

### 调试模式

如果遇到问题，可以设置Wandb为调试模式：

```bash
export WANDB_MODE=debug
```

这将提供更详细的日志信息来帮助诊断问题。
