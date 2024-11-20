"""
>  python query.py --tp 2 --pp 4 --world_size 16
tp: 2
pp: 4
dp: 2
world_size: 16
=========== TP Groups ============
[0, 1]
[2, 3]
[4, 5]
[6, 7]
[8, 9]
[10, 11]
[12, 13]
[14, 15]
=========== DP Groups ============
[0, 2]
[1, 3]
[4, 6]
[5, 7]
[8, 10]
[9, 11]
[12, 14]
[13, 15]
=========== PP Groups ============
[0, 4, 8, 12]
[1, 5, 9, 13]
[2, 6, 10, 14]
[3, 7, 11, 15]
"""


import argparse

def main():
    parser = argparse.ArgumentParser(description="Parse command line arguments for tp, pp, dp, and world_size.")
    parser.add_argument('--tp', type=int, help='Tensor parallelism size')
    parser.add_argument('--pp', type=int, help='Pipeline parallelism size')
    parser.add_argument('--world_size', type=int, help='Total world size')
    args = parser.parse_args()

    print(f"tp: {args.tp}")
    print(f"pp: {args.pp}")
    dp = args.world_size // (args.tp * args.pp)
    print(f"dp: {dp}")
    print(f"world_size: {args.world_size}")
    
    split(args.tp, args.pp, dp, args.world_size)

def pretty_print_groups(groups:list, name: str):
    print(f"=========== {name} ============")
    for g in groups:
        assert isinstance(g, list)
        s = ",\t".join([str(e) for e in g])
        print(g)
    

def split(tp, pp, dp, world_size):
    rank_per_pprank = world_size // pp
    
    tp_groups = []
    dp_groups = []
    for pp_i in range(pp):
        pp_rank_start = pp_i * rank_per_pprank
        
        tp_groups_of_this_pp_rank = []
        dp_groups_of_this_pp_rank = []
        
        #pp rank内部继续划分，tp连续
        for tp_start in range(pp_rank_start, pp_rank_start + rank_per_pprank, tp):
            tp_groups_of_this_pp_rank.append(
                list(range(tp_start, tp_start + tp))
            )
            
        for j in range(tp):
            dp_group = [tp_group[j] for tp_group in tp_groups_of_this_pp_rank]
            dp_groups_of_this_pp_rank.append(dp_group)
        
        tp_groups.extend(tp_groups_of_this_pp_rank)
        dp_groups.extend(dp_groups_of_this_pp_rank)
        
        
            
    pretty_print_groups(tp_groups, "TP Groups")
    pretty_print_groups(dp_groups, "DP Groups")
    
    pp_groups = []
    for pp_group_i in range(rank_per_pprank):
        pp_groups.append(list(range(pp_group_i, world_size, rank_per_pprank)))
        
    pretty_print_groups(pp_groups, "PP Groups")
    

if __name__ == "__main__":
    main()