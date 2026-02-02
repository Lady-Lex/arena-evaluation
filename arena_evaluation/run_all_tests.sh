#!/bin/bash

# 批量执行所有子文件夹的get_metrics命令
# 自动发现并执行data/tvss_nav/下所有task目录中的所有测试

echo "开始执行所有子文件夹的metrics计算..."

# 统计变量
total_tests=0
successful_tests=0
failed_tests=0

sub_dir_name="tvss_nav"
# sub_dir_name="vlm_nav"
# sub_dir_name="vlm_social_nav"


# 遍历所有task目录
for task_dir in data/$sub_dir_name/task_*; do
    if [ -d "$task_dir" ]; then
        task_name=$(basename "$task_dir")
        echo "正在处理 $task_name..."
        
        # 遍历该task目录下的所有测试目录
        for test_dir in "$task_dir"/task_*_test_*; do
            if [ -d "$test_dir" ]; then
                test_name=$(basename "$test_dir")
                data_dir="$test_dir/jackal"
                
                # 检查jackal子目录是否存在
                if [ -d "$data_dir" ]; then
                    total_tests=$((total_tests + 1))
                    echo "正在执行 $test_name..."
                    
                    # 根据任务类型选择不同的世界和参数
                    if [[ "$task_name" =~ ^task_[1-3]$ ]]; then
                        # Task 1-3: 使用 arena_hospital_small，需要 subject 和 zone 分数
                        world="arena_hospital_small"
                        params="--pedsim --subject --zone"
                        echo "执行: python scripts/get_metrics $data_dir $world $params"
                        python scripts/get_metrics "$data_dir" "$world" $params
                    elif [[ "$task_name" =~ ^task_[4-6]$ ]]; then
                        # Task 4-6: 使用 small_warehouse，不需要 subject 分数
                        world="small_warehouse"
                        params="--pedsim --subject --zone"
                        echo "执行: python scripts/get_metrics $data_dir $world $params"
                        python scripts/get_metrics "$data_dir" "$world" $params
                    fi
                    
                    if [ $? -eq 0 ]; then
                        echo "$test_name 执行成功!"
                        successful_tests=$((successful_tests + 1))
                    else
                        echo "$test_name 执行失败!"
                        failed_tests=$((failed_tests + 1))
                    fi
                else
                    echo "警告: 数据目录 $data_dir 不存在，跳过 $test_name"
                fi
                
                echo "----------------------------------------"
            fi
        done
    fi
done

echo "========================================"
echo "所有测试执行完成!"
echo "总测试数: $total_tests"
echo "成功: $successful_tests"
echo "失败: $failed_tests"
echo "========================================"
