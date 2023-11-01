from typing import List

def has_duplicates(int_list: List[int]) -> bool:
    seen = set()
    for num in int_list:
        if num in seen:
            return True
        seen.add(num)
    return False

def has_negatives(int_list: List[int]) -> bool:
    for num in int_list:
        if num < 0:
            return True
    return False