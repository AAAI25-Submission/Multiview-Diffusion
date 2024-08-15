#!/bin/bash

# 读取输入
method=$1
task=$2

# 使用 `=` 而不是 `==` 进行字符串比较
if [ "$method" = "eff" ]; then
    if [ "$task" = "generalization" ]; then
        cd ./Baselines/EffNet/ || exit 1
        python downStream_train.py --mode=test
    elif [ "$task" = "personalization" ]; then
        cd ./Baselines/EffNet/ || exit 1
        python per_train.py --mode=test --split_num=1
    else
        echo "Invalid task"
    fi
elif [ "$method" = "cmsc" ]; then
    if [ "$task" = "generalization" ]; then
        cd ./Baselines/CMSC/ || exit 1
        python ger_train.py --mode=test
    elif [ "$task" = "personalization" ]; then
        cd ./Baselines/CMSC/ || exit 1
        python per_train.py --mode=test --split_num=1
    else
        echo "Invalid task"
    fi
elif [ "$method" = "multiDiff" ]; then
    if [ "$task" = "generalization" ]; then
        cd ./Multiview/DiffusionModels/ || exit 1
        python ger_train.py --mode=test
    elif [ "$task" = "personalization" ]; then
        cd ./Multiview/DiffusionModels/ || exit 1
        python per_bayes_delta_train.py --mode=test --split_num=1
    else
        echo "Invalid task"
    fi
else
    echo "Invalid method"
fi