import torch
from graphviz import Digraph
import torch.nn as nn
import torch.nn.functional as F


node_id = 0


def add_node(G: Digraph, input_grad_fn, gradfn_to_viznode:dict,):
    if input_grad_fn in gradfn_to_viznode:
        return
    global node_id
    gradfn_to_viznode[input_grad_fn] = str(node_id)
    node_id += 1
    print(f"adding {str(type(input_grad_fn))=}")
    if type(input_grad_fn).__name__ == "AccumulateGrad":
        v = input_grad_fn.variable
        G.node(gradfn_to_viznode[input_grad_fn], label=f"AccumulateGrad:{list(v.shape)},{str(v.dtype)}", shape='rectangle', style='filled',)
    else:
        G.node(gradfn_to_viznode[input_grad_fn], label=f"{input_grad_fn.name()}")
    


def _viz_graph(G: Digraph, grad_fn, gradfn_to_viznode: dict, visited: set):
    
    if grad_fn is None:
        print(f"grad_fn is None")
        return
    
    if grad_fn in visited:
        return
    visited.add(grad_fn)
    
    for (input_grad_fn, index) in grad_fn.next_functions:
        # 创建节点
        #input_grad_fn 有3种情况：
        # 1. AccumulateGrad
        # 2. grad_fn比如AddBackward0
        # 3. None
        if input_grad_fn is None:
            continue
        
        add_node(G, input_grad_fn, gradfn_to_viznode)
        
        #创建边
        G.edge(head_name=gradfn_to_viznode[grad_fn], tail_name=gradfn_to_viznode[input_grad_fn], label=str(index))
        
        #如果不是AccumulateGrad，则递归处理
        if type(input_grad_fn).__name__ == "AccumulateGrad":       
            continue
        _viz_graph(G, input_grad_fn, gradfn_to_viznode, visited)


def viz_graph(t: torch.Tensor) -> Digraph:
    G = Digraph()
    if t.grad_fn is not None:
        global node_id
        G.node("out", label=f"out:{list(t.shape)},{str(t.dtype)}", shape='rectangle', style='filled', )
        gradfn_to_viznode = {}
        add_node(G, t.grad_fn, gradfn_to_viznode)
        G.edge(head_name="out", tail_name=gradfn_to_viznode[t.grad_fn])
        _viz_graph(G, t.grad_fn, gradfn_to_viznode, set())
    else:
        assert t.grad_fn is not None, f"{t=} should have a grad_fn"

    return G
