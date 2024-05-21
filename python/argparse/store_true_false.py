import argparse

# 创建 ArgumentParser 对象
parser = argparse.ArgumentParser(description="Example of 'store_true' action.")

# 添加带有 'store_true' 动作的参数
parser.add_argument('--tp-gemm-parallel', action='store_true',
                    help='Example of store_true action.')
parser.add_argument('--no-seq-par', action='store_false',
                    help='Example of store_false action.', dest='seq_par')

# 解析传递给程序的命令行参数
args = parser.parse_args()

# 打印参数值
print(f"The value of --tp-gemm-parallel is: {args.tp_gemm_parallel}")
print(f"The value of --seq-par is: {args.seq_par}")
