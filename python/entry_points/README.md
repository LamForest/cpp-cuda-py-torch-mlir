## 版本：
python 3.8

## 1. 安装entry points
```bash
pip install -e .
```

## 2. 加载entry points，并使用
详见 `use_entry_points.py``

```bash
╰─ python use_entry_points.py
hello from +  /home/github/cpp-cuda-py-torch-mlir/python/entry_points/src/pyproject_example/hello.py
```

## 参考资料：
1. [how to use entry points](https://stackoverflow.com/a/9615473)
2. [how to use entry points in pyproject.toml](https://stackoverflow.com/a/75419857), [官方文档](https://packaging.python.org/en/latest/specifications/pyproject-toml/#entry-points)