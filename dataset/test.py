import os
import numpy as np
import matplotlib.pyplot as plt

def plot_line_counts(line_counts):
    # 定义行数范围
    bins = range(0, max(line_counts) + 100, 100)
    hist, bin_edges = np.histogram(line_counts, bins=bins)
    plt.figure(figsize=(10, 6))  # 设置图形的大小
    plt.bar(bin_edges[:-1], hist, width=100)  # 绘制柱状图
    plt.xlabel('Line Count Ranges')  # 设置 x 轴标签
    plt.ylabel('Frequency')  # 设置 y 轴标签
    plt.title('Verilog File Line Count Distribution')  # 设置标题
    plt.show()
    # plt.tight_layout()  # 调整布局以确保标签完整显示
    plt.savefig('verilog_line_count_distribution.png')
    plt.close()

if __name__ == "__main__":
    # 请将 'your_directory_path' 替换为你存储 Verilog 文件的实际目录路径
    count = [206, 62, 192, 1292, 127, 117, 99, 152, 197, 254, 292, 363, 1837, 319, 549, 1623, 168, 155, 3011, 479, 601, 387, 986, 1547, 252, 242, 226, 30, 799, 113, 188, 522, 248, 74, 695, 379, 495, 479, 418, 844, 76, 299, 801, 783, 540, 824, 286, 163, 275, 345, 461, 90, 299, 92, 50, 46, 30, 57, 134, 50, 36, 120, 132, 198, 134, 42, 57, 269, 56, 133, 58, 278, 87, 246, 347, 175, 26, 32, 95, 31, 61, 315, 74, 112, 31, 98, 78, 173, 61, 53, 34, 38, 301, 128, 49, 154, 43, 77, 95, 26, 49, 68, 50, 161, 43, 66, 23, 60, 22, 55, 21, 40, 82, 46, 36, 264, 191, 83, 61, 64, 55, 79, 211, 57, 274, 89]
    print(count)
    plot_line_counts(count)

