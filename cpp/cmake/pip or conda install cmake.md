
先创建一个完全干净的conda环境
```bash
conda create -n tmp python=3.11
```

或者卸载一下当前conda环境的cmake：
```bash
pip uninstall conda
conda uninstall conda
``````

此时cmake用的是conda预置的：
```
(tmp) root@f3393221083d:/workspace# which cmake
/opt/conda/bin/cmake
```


然后通过pip安装cmake：
```
(tmp) root@f3393221083d:/workspace# pip install cmake
Collecting cmake
  Using cached cmake-3.28.3-py2.py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (6.3 kB)
Using cached cmake-3.28.3-py2.py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (26.3 MB)
Installing collected packages: cmake
Successfully installed cmake-3.28.3
WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv
(tmp) root@f3393221083d:/workspace# which cmake
/opt/conda/envs/tmp/bin/cmake
(tmp) root@f3393221083d:/workspace# cat `which cmake`
#!/opt/conda/envs/tmp/bin/python
# -*- coding: utf-8 -*-
import re
import sys
from cmake import cmake
if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
    sys.exit(cmake())
(tmp) root@f3393221083d:/workspace# python -c "import cmake"
```
通过conda pip分别观察cmake信息：
```
(tmp) root@f3393221083d:/workspace# pip show cmake
Name: cmake
Version: 3.28.3
Summary: CMake is an open-source, cross-platform family of tools designed to build, test and package software
Home-page: https://cmake.org/
Author: Jean-Christophe Fillion-Robin
Author-email: jchris.fillionr@kitware.com
License: Apache 2.0
Location: /opt/conda/envs/tmp/lib/python3.11/site-packages
Requires: 
Required-by: 
(tmp) root@f3393221083d:/workspace# conda list cmake
# packages in environment at /opt/conda/envs/tmp:
#
# Name                    Version                   Build  Channel
cmake                     3.28.3                   pypi_0    pypi
```
虽然是pip安装的，但是conda也能找到，注意Channel是pypi。

接下来，我们把pip安装的cmake卸载掉，通过conda安装：
```
(tmp) root@f3393221083d:/workspace# pip uninstall cmake 
Found existing installation: cmake 3.28.3
Uninstalling cmake-3.28.3:
  Would remove:
    /opt/conda/envs/tmp/bin/cmake
    /opt/conda/envs/tmp/bin/cpack
    /opt/conda/envs/tmp/bin/ctest
    /opt/conda/envs/tmp/lib/python3.11/site-packages/cmake-3.28.3.dist-info/*
    /opt/conda/envs/tmp/lib/python3.11/site-packages/cmake/*
Proceed (Y/n)? Y
  Successfully uninstalled cmake-3.28.3
WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv
(tmp) root@f3393221083d:/workspace# conda list cmake
# packages in environment at /opt/conda/envs/tmp:
#
# Name                    Version                   Build  Channel
(tmp) root@f3393221083d:/workspace# pip show cmake
WARNING: Package(s) not found: cmake

(tmp) root@f3393221083d:/workspace# conda install cmake
Collecting package metadata (current_repodata.json): done
Solving environment: done

## Package Plan ##

  environment location: /opt/conda/envs/tmp

  added / updated specs:
    - cmake


The following NEW packages will be INSTALLED:

  c-ares             pkgs/main/linux-64::c-ares-1.19.1-h5eee18b_0 
  cmake              pkgs/main/linux-64::cmake-3.26.4-h96355d8_0 
  expat              pkgs/main/linux-64::expat-2.5.0-h6a678d5_0 
  krb5               pkgs/main/linux-64::krb5-1.20.1-h143b758_1 
  libcurl            pkgs/main/linux-64::libcurl-8.5.0-h251f7ec_0 
  libedit            pkgs/main/linux-64::libedit-3.1.20230828-h5eee18b_0 
  libev              pkgs/main/linux-64::libev-4.33-h7f8727e_1 
  libnghttp2         pkgs/main/linux-64::libnghttp2-1.57.0-h2d74bed_0 
  libssh2            pkgs/main/linux-64::libssh2-1.10.0-hdbd6064_2 
  libuv              pkgs/main/linux-64::libuv-1.44.2-h5eee18b_0 
  lz4-c              pkgs/main/linux-64::lz4-c-1.9.4-h6a678d5_0 
  rhash              pkgs/main/linux-64::rhash-1.4.3-hdbd6064_0 
  zstd               pkgs/main/linux-64::zstd-1.5.5-hc292b87_0 


Proceed ([y]/n)? y


Downloading and Extracting Packages:

Preparing transaction: done
Verifying transaction: done
Executing transaction: done
(tmp) root@f3393221083d:/workspace# which cmake
/opt/conda/envs/tmp/bin/cmake
(tmp) root@f3393221083d:/workspace# ll -h `which cmake`
-rwxr-xr-x 2 root root 12M Jun 16  2023 /opt/conda/envs/tmp/bin/cmake*
```
通过conda，可以观测到安装的cmake，注意，这里Channel不再是pypi：
```
(tmp) root@f3393221083d:/workspace# conda list cmak
# packages in environment at /opt/conda/envs/tmp:
#
# Name                    Version                   Build  Channel
cmake                     3.26.4               h96355d8_0 
```
但是通过pip不行，也不能在python中import
```
(tmp) root@f3393221083d:/workspace# pip show cmake
WARNING: Package(s) not found: cmake
(tmp) root@f3393221083d:/workspace# python -c "import cmake"
Traceback (most recent call last):
  File "<string>", line 1, in <module>
ModuleNotFoundError: No module named 'cmake'
```
这说明conda安装的事cmake二进制，而不是python包。


此时，再通过pip安装cmake，不会报错，但是会覆盖conda安装的cmake，又变成了通过脚本调用cmake：
```
(tmp) root@f3393221083d:/workspace# cat `which cmake`
#!/opt/conda/envs/tmp/bin/python
# -*- coding: utf-8 -*-
import re
import sys
from cmake import cmake
if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
    sys.exit(cmake())
```

再通过conda和pip观察，已经看不到conda安装的痕迹了：
```
(tmp) root@f3393221083d:/workspace# conda list cmake
# packages in environment at /opt/conda/envs/tmp:
#
# Name                    Version                   Build  Channel
cmake                     3.28.3                   pypi_0    pypi
(tmp) root@f3393221083d:/workspace# pip show cmake
Name: cmake
Version: 3.28.3
Summary: CMake is an open-source, cross-platform family of tools designed to build, test and package software
Home-page: https://cmake.org/
Author: Jean-Christophe Fillion-Robin
Author-email: jchris.fillionr@kitware.com
License: Apache 2.0
Location: /opt/conda/envs/tmp/lib/python3.11/site-packages
Requires: 
Required-by:
```

再通过conda uninstall cmake卸载，此时，tmp环境中再也找不到cmake了，说明conda uninstall的时候，由于pip 和 conda安装的cmake都位于/opt/conda/envs/tmp/bin/cmake，所以被conda卸载了。
```
(tmp) root@f3393221083d:/workspace# which cmake
/opt/conda/bin/cmake
```
此时pip安装的cmake处在一个不完整的状态，存在python包，但是PATH中没有二进制文件：
```
(tmp) root@f3393221083d:/workspace# which cmake
/opt/conda/bin/cmake
(tmp) root@f3393221083d:/workspace# python -c "import cmake"
(tmp) root@f3393221083d:/workspace# pip show cmake
Name: cmake
Version: 3.28.3
Summary: CMake is an open-source, cross-platform family of tools designed to build, test and package software
Home-page: https://cmake.org/
Author: Jean-Christophe Fillion-Robin
Author-email: jchris.fillionr@kitware.com
License: Apache 2.0
Location: /opt/conda/envs/tmp/lib/python3.11/site-packages
Requires: 
Required-by: 
(tmp) root@f3393221083d:/workspace# conda list cmake
# packages in environment at /opt/conda/envs/tmp:
#
# Name                    Version                   Build  Channel
cmake                     3.28.3                   pypi_0    pypi
(tmp) root@f3393221083d:/workspace# cmake --version
cmake version 3.26.4 #不是3.28.3
```




## 结论
pip安装的cmake和conda安装的cmake是不一样的，pip安装的cmake是python包，而conda安装的cmake是编译好的二进制文件。

如果都装了，可能有问题？要保证都卸载干净后再继续。