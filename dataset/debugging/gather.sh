#!/bin/bash

# 设置目标目录名称
target_dir="database"

# 创建目标目录（如果不存在）
mkdir -p "$target_dir"

# 查找并复制所有 .v 和 .sv 文件（仅当前目录）
find . -type f \( -name "*.v" -o -name "*.sv" \) -exec cp {} "$target_dir" \;

echo "已复制所有 Verilog 和 SystemVerilog 文件到 $target_dir 目录"
