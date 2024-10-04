
以前还真没注意过nn.Module的实现。。。

当在module中进行以下操作：
- self.fc = nn.Linear()的时候，实际上是在王self._modules['fc']=nn.Linear()
- self.weight = nn.Parameter()，实际上是self.register_parameters('weight', nn.Parameter())，即self._parameters['weight']=nn.Parameter()

当通过model.fc访问时，nn.Module通过`__getattr__`拦截，并返回self._modules['fc']。注意到`__getattr__`存在优先顺序。

当通过delattr(model, 'fc') 时，等价于 del model._modules['fc']。


当通过 `named_modules`遍历时，实际上是**深度优先遍历** `self._modules`，我们写个非常简单的程序验证下：
```py
#traverse.py
import torch
import torch.nn as nn

# 定义一个简单的嵌套神经网络模块
class NestedModule(nn.Module):
    def __init__(self):
        super(NestedModule, self).__init__()
        # 第一层
        self.layer1 = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU()
        )
        # 第二层，包含另一个模块
        self.layer2 = nn.Sequential(
            nn.Linear(20, 30),
            nn.ReLU(),
            nn.Sequential(  # 更深的嵌套
                nn.Linear(30, 40),
                nn.ReLU()
            )
        )
        # 第三层
        self.layer3 = nn.Linear(40, 50)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

# 创建嵌套模块的实例
model = NestedModule()

# 使用named_modules遍历所有模块
for name, module in model.named_modules():
    print(name)


"""
#输出为:
layer1
layer1.0
layer1.1
layer2
layer2.0
layer2.1
layer2.2
layer2.2.0
layer2.2.1
layer3
#确实是深度优先遍历
"""
```

```py
    # On the return type:
    # We choose to return `Any` in the `__getattr__` type signature instead of a more strict `Union[Tensor, Module]`.
    # This is done for better interop with various type checkers for the end users.
    # Having a stricter return type doesn't play nicely with `register_buffer()` and forces
    # people to excessively use type-ignores, asserts, casts, etc.
    # See full discussion on the problems with returning `Union` here
    # https://github.com/microsoft/pyright/issues/4213
    def __getattr__(self, name: str) -> Any:
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return _parameters[name]
        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers:
                return _buffers[name]
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return modules[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Union[Tensor, 'Module']) -> None:
        def remove_from(*dicts_or_sets):
            for d in dicts_or_sets:
                if name in d:
                    if isinstance(d, dict):
                        del d[name]
                    else:
                        d.discard(name)

        params = self.__dict__.get('_parameters')
        if isinstance(value, Parameter):
            if params is None:
                raise AttributeError(
                    "cannot assign parameters before Module.__init__() call")
            remove_from(self.__dict__, self._buffers, self._modules, self._non_persistent_buffers_set)
            self.register_parameter(name, value)
        elif params is not None and name in params:
            if value is not None:
                raise TypeError(f"cannot assign '{torch.typename(value)}' as parameter '{name}' "
                                "(torch.nn.Parameter or None expected)"
                                )
            self.register_parameter(name, value)
        else:
            modules = self.__dict__.get('_modules')
            if isinstance(value, Module):
                if modules is None:
                    raise AttributeError(
                        "cannot assign module before Module.__init__() call")
                remove_from(self.__dict__, self._parameters, self._buffers, self._non_persistent_buffers_set)
                for hook in _global_module_registration_hooks.values():
                    output = hook(self, name, value)
                    if output is not None:
                        value = output
                modules[name] = value
            elif modules is not None and name in modules:
                if value is not None:
                    raise TypeError(f"cannot assign '{torch.typename(value)}' as child module '{name}' "
                                    "(torch.nn.Module or None expected)"
                                    )
                for hook in _global_module_registration_hooks.values():
                    output = hook(self, name, value)
                    if output is not None:
                        value = output
                modules[name] = value
            else:
                buffers = self.__dict__.get('_buffers')
                if buffers is not None and name in buffers:
                    if value is not None and not isinstance(value, torch.Tensor):
                        raise TypeError(f"cannot assign '{torch.typename(value)}' as buffer '{name}' "
                                        "(torch.Tensor or None expected)"
                                        )
                    for hook in _global_buffer_registration_hooks.values():
                        output = hook(self, name, value)
                        if output is not None:
                            value = output
                    buffers[name] = value
                else:
                    super().__setattr__(name, value)

    def __delattr__(self, name):
        if name in self._parameters:
            del self._parameters[name]
        elif name in self._buffers:
            del self._buffers[name]
            self._non_persistent_buffers_set.discard(name)
        elif name in self._modules:
            del self._modules[name]
        else:
            super().__delattr__(name)


    def register_parameter(self, name: str, param: Optional[Parameter]) -> None:
        r"""Adds a parameter to the module.

        The parameter can be accessed as an attribute using given name.

        Args:
            name (str): name of the parameter. The parameter can be accessed
                from this module using the given name
            param (Parameter or None): parameter to be added to the module. If
                ``None``, then operations that run on parameters, such as :attr:`cuda`,
                are ignored. If ``None``, the parameter is **not** included in the
                module's :attr:`state_dict`.
        """
        if '_parameters' not in self.__dict__:
            raise AttributeError(
                "cannot assign parameter before Module.__init__() call")

        elif not isinstance(name, str):
            raise TypeError(f"parameter name should be a string. Got {torch.typename(name)}")
        elif '.' in name:
            raise KeyError("parameter name can't contain \".\"")
        elif name == '':
            raise KeyError("parameter name can't be empty string \"\"")
        elif hasattr(self, name) and name not in self._parameters:
            raise KeyError(f"attribute '{name}' already exists")

        if param is None:
            self._parameters[name] = None
        elif not isinstance(param, Parameter):
            raise TypeError(f"cannot assign '{torch.typename(param)}' object to parameter '{name}' "
                            "(torch.nn.Parameter or None required)"
                            )
        elif param.grad_fn:
            raise ValueError(
                f"Cannot assign non-leaf Tensor to parameter '{name}'. Model "
                f"parameters must be created explicitly. To express '{name}' "
                "as a function of another Tensor, compute the value in "
                "the forward() method.")
        else:
            for hook in _global_parameter_registration_hooks.values():
                output = hook(self, name, param)
                if output is not None:
                    param = output
            self._parameters[name] = param

    def named_modules(self, memo: Optional[Set['Module']] = None, prefix: str = '', remove_duplicate: bool = True):
        r"""Returns an iterator over all modules in the network, yielding
        both the name of the module as well as the module itself.

        Args:
            memo: a memo to store the set of modules already added to the result
            prefix: a prefix that will be added to the name of the module
            remove_duplicate: whether to remove the duplicated module instances in the result
                or not

        Yields:
            (str, Module): Tuple of name and module

        Note:
            Duplicate modules are returned only once. In the following
            example, ``l`` will be returned only once.

        Example::

            >>> l = nn.Linear(2, 2)
            >>> net = nn.Sequential(l, l)
            >>> for idx, m in enumerate(net.named_modules()):
            ...     print(idx, '->', m)

            0 -> ('', Sequential(
              (0): Linear(in_features=2, out_features=2, bias=True)
              (1): Linear(in_features=2, out_features=2, bias=True)
            ))
            1 -> ('0', Linear(in_features=2, out_features=2, bias=True))

        """

        if memo is None:
            memo = set()
        if self not in memo:
            if remove_duplicate:
                memo.add(self)
            yield prefix, self
            for name, module in self._modules.items():
                if module is None:
                    continue
                submodule_prefix = prefix + ('.' if prefix else '') + name
                yield from module.named_modules(memo, submodule_prefix, remove_duplicate)
```