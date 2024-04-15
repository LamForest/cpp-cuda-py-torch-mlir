from tabulate import tabulate
# 定义红色文字的ANSI转义码
RED = "\033[1;31m"
# 定义重置颜色的ANSI转义码
RESET = "\033[0m"


def process_line(line):
    """处理单行，如果符合格式则返回参数名和值，否则返回None。"""
    line = line.strip()
    if line.startswith("#"):
        return None
    if line.endswith('"'):
        line = line[:-1].strip()
        
    if line.endswith("\\"):
        line = line[:-1].strip()
    if line.startswith("--"):
        line = line[2:]
        parts = line.split(" ")
        if len(parts) == 2:
            param_name = parts[0]
            param_value = parts[1]
            assert param_value, f"{param_value} should have a value"
            return param_name, param_value
        elif len(parts) == 1:
            param_name = parts[0]
            param_value = ""
            return param_name, param_value
            
    return None

def read_file(filename):
    """读取文件并返回参数字典。"""
    params = {}
    with open(filename, 'r') as file:
        for line in file:
            result = process_line(line)
            if result:
                param_name, param_value = result
                params[param_name] = param_value
    return params

def compare_dicts(dict1, dict2, file1, file2):
    """比较两个字典并打印结果。"""
    keys1 = set(dict1.keys())
    keys2 = set(dict2.keys())
    
    shared_keys = keys1.intersection(keys2)
    unique_to_file1 = keys1 - keys2
    unique_to_file2 = keys2 - keys1
    
    shared_params = [[key, dict1[key], dict2[key]] for key in shared_keys]
    for i, params in enumerate(shared_params):
        k, v1, v2 = params
        if v1 != v2:
            v2 = f"{RED}{v2}{RESET}"
        shared_params[i] = [k, v1, v2]      
    unique_params1 = [[key, dict1[key]] for key in unique_to_file1]
    unique_params2 = [[key, dict2[key]] for key in unique_to_file2]
    
    print("Parameters shared by both files:")
    print(tabulate(shared_params, headers=['Parameter', f'File: {file1}', f'File: {file2}'], tablefmt='fancy_grid'))
    
    print(f"\nParameters unique to file 1: {file1}")
    print(tabulate(unique_params1, headers=['Parameter', 'Value'], tablefmt='fancy_grid'))
    
    print(f"\nParameters unique to file 2: {file2}")
    print(tabulate(unique_params2, headers=['Parameter', 'Value'], tablefmt='fancy_grid'))


# 主程序
def main(file1, file2):
    dict1 = read_file(file1)
    dict2 = read_file(file2)
    compare_dicts(dict1, dict2, file1, file2)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python script.py <file1> <file2>")
    else:
        file1, file2 = sys.argv[1], sys.argv[2]
        main(file1, file2)