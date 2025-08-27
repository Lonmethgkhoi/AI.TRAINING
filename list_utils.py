def flatten(lst):
    """Làm phẳng danh sách 2 chiều"""
    return [item for sublist in lst for item in sublist]

def unique(lst):
    """Trả về danh sách các phần tử duy nhất"""
    return list(set(lst))

def sort_list(lst):
    """Sắp xếp danh sách"""
    return sorted(lst)

def merge_lists(lst1, lst2):
    """Gộp 2 danh sách"""
    return lst1 + lst2

def filter_even(lst):
    """Lọc các số chẵn"""
    return [x for x in lst if x % 2 == 0]

def filter_odd(lst):
    """Lọc các số lẻ"""
    return [x for x in lst if x % 2 != 0]

def map_square(lst):
    """Bình phương từng phần tử"""
    return [x**2 for x in lst]

def sum_nested(lst):
    """Tổng các danh sách con"""
    return sum(sum(sub) for sub in lst)

def reverse_list(lst):
    """Đảo ngược danh sách"""
    return lst[::-1]
