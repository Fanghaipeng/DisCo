#!/bin/bash

# 定义要删除的子目录名称
declare -a subdirs=("cond" "gt" "gt_gif" "ref" "ref_control")

# 搜索并删除指定的子目录
find /data/fanghaipeng/project/MotionFlow/runtest/exp -type d -regex ".*/[^/]+/[^/]+$" | while read dir; do
    for subdir in "${subdirs[@]}"; do
        if [ -d "${dir}/${subdir}" ]; then
            echo "Deleting ${dir}/${subdir}"
            rm -rf "${dir}/${subdir}"
        fi
    done
done

echo "Deletion complete."
