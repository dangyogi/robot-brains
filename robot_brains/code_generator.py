# code_generator.py

import sys
import os.path
import operator
from functools import partial
from io import StringIO

from robot_brains import symtable
from robot_brains.scanner import Token, syntax_error


Num_label_bits = 1              # running bit
Max_bits_used = 0


def add_hook_args_to_todo(cls, hook_name, fn, list):
    cls.add_post_hook(hook_name, lambda *args: list.append(partial(fn, *args)))


def todo(list):
    def decorator(fn):
        list.append(fn)
        return fn
    return decorator


def todo_with_args(cls, hook_name, list):
    def decorator(fn):
        add_hook_args_to_todo(cls, hook_name, fn, list)
        return fn
    return decorator


def run_todo_lists():
    for list_name, list in Target_language.Todo_lists:
        print("Running", list_name)
        for fn in list:
            fn()


def init_code_generator(target_language, rootdir):
    global Target_language, Precedence_lookup, Rootdir

    Target_language = target_language
    Rootdir = rootdir

    # {operator: (precedence_level, assoc)}
    Precedence_lookup = parse_precedence(Target_language.Precedence)
    target_language.Precedence_lookup = Precedence_lookup


def relative_path(filename):
    return filename[len(Rootdir) + 1:]


def generate(opmode):
    Target_language.Opmode = opmode
    print("generate: running opmode.setup")
    opmode.setup()
    print()
    print("generate: running Todo_lists")
    run_todo_lists()
    print("generate: done")


def parse_precedence(precedence):
    return {op: (i, assoc)
            for i, (assoc, *opers) in enumerate(reversed(precedence), 1)
            for op in opers
           }


def wrap(expr, precedence, side):
    if expr.precedence > precedence or \
       expr.precedence == precedence and expr.assoc == side:
        return expr.code
    return f"({expr.code})"


def compile_unary(op, expr):
    return Target_language.Unary_exprs[op.upper()](expr)


def compile_binary(expr1, op, expr2):
    return Target_language.Binary_exprs[op.upper()](expr1, expr2)


def unary(op, format, expr): 
    precedence, assoc = Precedence_lookup[op]
    return (precedence, assoc,
            format.format(wrap(expr, precedence, 'right')))


def binary(op, format, left_expr, right_expr, *, result_op=None): 
    precedence, assoc = Precedence_lookup[op]
    if result_op is None:
        result_prec, result_assoc = precedence, assoc
    else:
        result_prec, result_assoc = Precedence_lookup(result_op)
    return (result_prec, result_assoc,
            format.format(wrap(left_expr, precedence, 'left'),
                          wrap(right_expr, precedence, 'right')))


def subscript(left, right):
    # range checking must be done somewhere else...
    precedence, assoc = Precedence_lookup['[']
    return (precedence, assoc,
            f"{wrap(left, precedence, 'left')}[{right.code}]")


def translate_name(name):
    if isinstance(name, Token):
        name = name.value
    if not name[-1].isdecimal() and not name[-1].isalpha():
        return name[:-1].lower() + '_'
    if name.lower() in Target_language.Reserved_words:
        return name.lower() + '_'
    return name.lower()


def translate_type(type):
    type = type.get_type()
    if isinstance(type, symtable.Builtin_type):
        return Target_language.Types[type.name.lower()]
    if isinstance(type, symtable.Label_type):
        return Target_language.translate_label_type(type)
    assert isinstance(type, symtable.Typename_type)
    return translate_type(type.typedef.type)

