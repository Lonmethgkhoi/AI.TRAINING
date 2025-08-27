def factorial_recursive(n):
    """Giai thừa đệ quy"""
    return 1 if n==0 else n*factorial_recursive(n-1)

def fibonacci_recursive(n):
    """Fibonacci đệ quy"""
    if n <= 1:
        return n
    return fibonacci_recursive(n-1) + fibonacci_recursive(n-2)

def sum_nested_list(lst):
    """Tổng danh sách lồng nhau đệ quy"""
    total = 0
    for item in lst:
        if isinstance(item, list):
            total += sum_nested_list(item)
        else:
            total += item
    return total

def hanoi(n, source='A', target='C', auxiliary='B'):
    """Tháp Hà Nội"""
    if n==1:
        print(f"Move disk 1 from {source} to {target}")
        return
    hanoi(n-1, source, auxiliary, target)
    print(f"Move disk {n} from {source} to {target}")
    hanoi(n-1, auxiliary, target, source)
