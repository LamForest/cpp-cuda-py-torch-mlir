import torch
import sys
sys.path.append("../torchviz")
from viz import viz_graph

class XrayBackwardHook(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args):
        ctx.mark_non_differentiable(*[arg for arg in args if not arg.requires_grad])
        return args

    @staticmethod
    def backward(ctx, *args):
        return args

def apply_hook(args, fn):
    tensors = []
    tensors_idx = []
    requires_grad = False
    
    for i, arg in enumerate(args):
        if isinstance(arg, torch.Tensor):
            tensors.append(arg)
            tensors_idx.append(i)
            requires_grad |= arg.requires_grad

    if not (requires_grad and torch.is_grad_enabled()):
        return args
    
    new_tensors = XrayBackwardHook.apply(*tensors)

    
    assert len(new_tensors) > 0
    grad_fns = [t.grad_fn for t in new_tensors if t.grad_fn is not None and t.grad_fn.name() == "XrayBackwardHookBackward"]
    
    print(f"{grad_fns=}")
    
    grad_fns[0].register_hook(fn)

    arg_list = list(args)
    for idx, val in zip(tensors_idx, new_tensors):
        arg_list[idx] = val
    return tuple(arg_list)
    
def backward_after_hook(*args, **kwargs):
    print(f"backward_after_hook")
def backward_pre_hook(*args, **kwargs):
    print(f"backward_pre_hook")
    
def begin_tag(*args, tag:str):
    args = apply_hook(args, fn=lambda *_a, **_b: print(f'[BWD END {tag}]'))
    if len(args) == 1: #简单粗暴，先这么用吧
        return args[0]
    return args

def end_tag(ret, tag:str):
    is_tuple = True
    if not isinstance(ret, tuple):
        ret = (ret,)
        is_tuple = False
    ret = apply_hook(ret, fn=lambda *_a, **_b: print(f'[BWD BEGIN {tag}]'))
    if not is_tuple:
        ret = ret[0]
    return ret


def xray_tag(tag="unknown tag"):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if kwargs is None:
                kwargs = {}
            assert len(kwargs) == 0, f"暂不支持kwargs hook"
            args = begin_tag(*args, tag=tag)
            if not isinstance(args, tuple):
                args = (args,)
            print(f"[FWD BEGIN {tag}]")
            ret = func(*args, **kwargs)
            print(f"[FWD END {tag}]")
            
            ret = end_tag(ret, tag=tag)
            return ret
        return wrapper
    return decorator


##################测试##################

x1, x2 = (
    torch.randn(2, 3, requires_grad=True),
    torch.randn(2, 3, requires_grad=True),
)
x = torch.randn(2, 3, requires_grad=True)

@xray_tag("fff")
def f(x, y):
    out = (x -y).mul(5)
    return out


def f1(x, y):
    x, y = begin_tag(x, y, tag="fff")
    out = (x -y).mul(5)
    out = end_tag(out, tag="fff")
    return out

@xray_tag("ff2")
def f2(x):
    return x.sub(5).div(2)


out = f1(x1, x2)

viz_graph(out).render("test", format="png")
out.sum().backward()


out = f2(x)
viz_graph(out).render("test2", format="png")
out.sum().backward()
