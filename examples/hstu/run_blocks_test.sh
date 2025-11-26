# #!/bin/bash
# # run_blocks_test.sh
# # 测试不同 blocks_in_primary_pool 参数的性能

# BASE_CMD="python3 ./inference/inference_gr_ranking_async.py \
#     --gin_config_file ./inference/configs/kuairand_1k_inference_ranking.gin \
#     --checkpoint_dir ./checkpoints/iter500 \
#     --mode simulate \
#     --max_bs 8 \
#     --gpu 0"

# # 定义参数范围
# # START=4096
# # END=40960
# START=43008
# END=43009
# STEP=2048

# # 记录失败参数的数组
# FAILED_BLOCKS=()

# echo "开始测试 blocks_in_primary_pool 参数从 ${START} 到 ${END}，步长 ${STEP}"
# echo "=========================================================="

# for ((blocks=$START; blocks<=$END; blocks+=$STEP)); do
#     echo "正在测试 blocks_in_primary_pool = $blocks"
#     echo "----------------------------------------"
    
#     # 执行命令，如果失败则记录并继续
#     if $BASE_CMD --blocks_in_primary_pool $blocks; then
#         echo "blocks_in_primary_pool = $blocks 测试完成"
#     else
#         echo "blocks_in_primary_pool = $blocks 测试失败"
#         FAILED_BLOCKS+=($blocks)
#     fi
    
#     echo "=========================================================="
# done

# echo "所有测试完成！"

# # 显示失败参数汇总
# if [ ${#FAILED_BLOCKS[@]} -eq 0 ]; then
#     echo "所有参数测试都成功！"
# else
#     echo "以下参数测试失败："
#     for block in "${FAILED_BLOCKS[@]}"; do
#         echo "   - blocks_in_primary_pool = $block"
#     done
#     echo "总共 ${#FAILED_BLOCKS[@]} 个参数测试失败"
# fi

BASE_CMD="python3 ./inference/inference_gr_ranking_async.py \
    --gin_config_file ./inference/configs/kuairand_1k_inference_ranking.gin \
    --checkpoint_dir ./checkpoints/iter500 \
    --mode simulate \
    --blocks_in_primary_pool 40960 \
    --gpu 0"

# 定义 max_bs 参数范围
# MAX_BS_VALUES=(1 2 4 8 12 16 32)
MAX_BS_VALUES=(12)

# 记录失败参数的数组
FAILED_BS=()

echo "开始测试 max_bs 参数：${MAX_BS_VALUES[@]}"
echo "=========================================================="

for max_bs in "${MAX_BS_VALUES[@]}"; do
    echo "正在测试 max_bs = $max_bs"
    echo "----------------------------------------"
    
    # 执行命令，如果失败则记录并继续
    if $BASE_CMD --max_bs $max_bs; then
        echo "max_bs = $max_bs 测试完成"
    else
        echo "max_bs = $max_bs 测试失败"
        FAILED_BS+=($max_bs)
    fi
    
    echo "=========================================================="
done

echo "所有测试完成！"

# 显示失败参数汇总
if [ ${#FAILED_BS[@]} -eq 0 ]; then
    echo "所有 max_bs 参数测试都成功！"
else
    echo "以下 max_bs 参数测试失败："
    for bs in "${FAILED_BS[@]}"; do
        echo "   - max_bs = $bs"
    done
    echo "总共 ${#FAILED_BS[@]} 个参数测试失败"
fi