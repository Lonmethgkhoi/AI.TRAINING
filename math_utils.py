def add(a, b):
    """Trả về a + b"""
    return a + b

def subtract(a, b):
    """Trả về a - b"""
    return a - b

def multiply(a, b):
    """Trả về a * b"""
    return a * b

def divide(a, b):
    """Trả về a / b hoặc None nếu b = 0"""
    if b != 0:
        return a / b
    return None

def factorial(n):
    """Tính giai thừa của n"""
    if n == 0:
        return 1
    return n * factorial(n-1)

def fibonacci(n):
    """Trả về số Fibonacci thứ n"""
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a+b
    return a

def gcd(a, b):
    """Ước chung lớn nhất của a và b"""
    while b:
        a, b = b, a % b
    return a

def lcm(a, b):
    """Bội chung nhỏ nhất của a và b"""
    return abs(a*b) // gcd(a, b)

def power(a, b):
    """a mũ b"""
    return a ** b

def is_prime(n):
    """Kiểm tra số nguyên tố"""
    if n < 2:
        return False
    for i in range(2, int(n**0.5)+1):
        if n % i == 0:
            return False
    return True
