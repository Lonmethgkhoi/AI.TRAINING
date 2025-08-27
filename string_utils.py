def reverse_string(s):
    """Đảo ngược chuỗi"""
    return s[::-1]

def is_palindrome(s):
    """Kiểm tra palindrome"""
    return s == s[::-1]

def count_vowels(s):
    """Đếm nguyên âm"""
    return sum(1 for c in s.lower() if c in "aeiou")

def count_consonants(s):
    """Đếm phụ âm"""
    return sum(1 for c in s.lower() if c.isalpha() and c not in "aeiou")

def to_upper(s):
    """Chuyển chuỗi thành chữ in hoa"""
    return s.upper()

def to_lower(s):
    """Chuyển chuỗi thành chữ thường"""
    return s.lower()

def capitalize_words(s):
    """Viết hoa chữ cái đầu mỗi từ"""
    return s.title()

def replace_spaces(s, char="_"):
    """Thay khoảng trắng bằng ký tự khác"""
    return s.replace(" ", char)

def remove_punctuation(s):
    """Xóa ký tự đặc biệt"""
    import string
    return s.translate(str.maketrans('', '', string.punctuation))
