import random
import string
from typing import List, Tuple


def is_well_formed_dyck(s: str, parens: List[Tuple[str, str]]) -> bool:
    open_brackets = list(map(lambda x: x[0], parens))
    stack = []
    
    for c in s:
        if c in open_brackets:
            stack.append(c)
        else:
            if stack and (stack[-1] in open_brackets) and ((stack[-1], c) in  parens):
                stack.pop()
            else:
                return False
    return not stack

def swap_paren(old_paren : str, parens: List[Tuple[str, str]]) -> str:
    # Trick to flatten a list of tuples
    flat = list(sum(parens, ()))
    new_paren = random.choice(flat)

    while new_paren == old_paren:
        new_paren = random.choice(flat)

    return new_paren

def deform(s : str, parens: List[Tuple[str, str]]) -> str:
    deformed_s = s
    while is_well_formed_dyck(s=deformed_s, parens=parens):
        deform_index = random.randint(0, len(s) - 1)
        new_paren = swap_paren(old_paren=s[deform_index], parens=parens)
        deformed_s = deformed_s[:deform_index] + [new_paren] + deformed_s[deform_index + 1:]

    return deformed_s
