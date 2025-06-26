#!/usr/bin/env python3
"""
批量运行find_duals.py脚本
用于收集大量对偶点以进行神经网络权重提取
"""

import subprocess
import sys
import time
import os


def run_find_dual_points(num_iterations=100):
    """
    批量运行find_duals.py

    Args:
        num_iterations: 运行次数，每次生成10,000个对偶点
    """
    print(f"开始批量运行find_duals.py，共{num_iterations}次迭代")
    print(f"预计将生成 {num_iterations * 10000:,} 个对偶点")
    print("-" * 50)

    success_count = 0
    fail_count = 0

    for i in range(num_iterations):
        try:
            print(f"运行第 {i+1}/{num_iterations} 次...")
            start_time = time.time()

            # 运行find_dual_points.py
            result = subprocess.run(
                [sys.executable, "find_duals.py"],
                capture_output=True,
                text=True,
                check=True,
            )

            end_time = time.time()
            duration = end_time - start_time
            success_count += 1

            print(f"✓ 第 {i+1} 次完成，耗时: {duration:.2f}秒")

            # 显示进度
            if (i + 1) % 10 == 0:
                total_points = (i + 1) * 10000
                print(f"进度: {i+1}/{num_iterations}, 已收集 {total_points:,} 个对偶点")
                print("-" * 30)

        except subprocess.CalledProcessError as e:
            fail_count += 1
            print(f"✗ 第 {i+1} 次失败: {e}")
            print(f"错误输出: {e.stderr}")

        except KeyboardInterrupt:
            print(f"\n用户中断操作，已完成 {success_count} 次运行")
            break

        except Exception as e:
            fail_count += 1
            print(f"✗ 第 {i+1} 次发生未知错误: {e}")

    # 总结
    print("\n" + "=" * 50)
    print("批量运行完成!")
    print(f"成功: {success_count} 次")
    print(f"失败: {fail_count} 次")
    print(f"总共收集约 {success_count * 10000:,} 个对偶点")
    print("=" * 50)


def main():
    # 检查find_dual_points.py是否存在
    if not os.path.exists("find_duals.py"):
        print("错误: 找不到find_duals.py文件")
        print("请确保您在正确的目录中运行此脚本")
        return

    # 获取用户输入的运行次数
    if len(sys.argv) > 1:
        try:
            num_iterations = int(sys.argv[1])
        except ValueError:
            print("错误: 请提供有效的数字作为运行次数")
            return
    else:
        try:
            num_iterations = int(input("请输入要运行的次数 (默认100): ") or "100")
        except ValueError:
            print("使用默认值: 100次")
            num_iterations = 100

    if num_iterations <= 0:
        print("错误: 运行次数必须大于0")
        return

    # 确认是否继续
    print(f"将运行find_duals.py {num_iterations}次")
    print(f"预计生成 {num_iterations * 10000:,} 个对偶点")
    confirm = input("是否继续? (y/N): ").lower().strip()

    if confirm in ["y", "yes", "是"]:
        run_find_dual_points(num_iterations)
    else:
        print("操作已取消")


if __name__ == "__main__":
    main()
