def merge_dicts(d1, d2):
    """Gộp 2 dict"""
    return {**d1, **d2}

def invert_dict(d):
    """Đảo key thành value"""
    return {v:k for k,v in d.items()}

def count_keys(d):
    """Đếm số key"""
    return len(d)

def group_by_value(lst):
    """Nhóm các phần tử theo giá trị"""
    from collections import defaultdict
    res = defaultdict(list)
    for key, val in lst:
        res[val].append(key)
    return dict(res)

def dict_keys_to_list(d):
    """Trả về danh sách key"""
    return list(d.keys())
