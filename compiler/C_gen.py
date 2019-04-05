# C_gen.py

from functools import partial

from symtable import indent_str


Struct_indent = 0
Struct_var_indent = 4
Label_indent = 2
Statement_indent = 4


precedence = (  # highest to lowest
    ('left', '.', '(', '[', '->'),
    ('right', 'UMINUS', '++', '--', '!', '~', '&', '&&', 'sizeof'),
    ('left', '*', '/', '%'),
    ('left', '+', '-'),
    ('left', '<<', '>>'),
    ('left', '<', '<=', '>', '>='),
    ('left', '==', '!='),
    ('left', '&'),
    ('left', '^'),
    ('left', '|'),
    ('left', '&&'),
    ('left', '||'),
    ('right', '?:'),
    ('right', '=', '*=', '/=', '%=', '-=', '&=', '^=', '|=', '<<=', '>>='),
    ('left', ','),
)


def parse_precedence(precedence):
    return {op: (i, assoc)
            for i, (assoc, *opers) in enumerate(reversed(precedence), 1)
            for op in opers
           }


# {operator: (precedence_level, assoc)}
Precedence_lookup = parse_precedence(precedence)


def wrap(expr, precedence, side):
    if expr[2] > precedence or expr[2] == precedence and expr[3] == side:
        return expr[1]
    return f"({expr[1]})"


def unary(op, format, expr): 
    precedence, assoc = Precedence_lookup[op]
    return (expr[0],
            format.format(wrap(expr, precedence, 'right')),
            precedence, assoc)


def binary(op, format, left_expr, right_expr, *, result_op=None): 
    precedence, assoc = Precedence_lookup[op]
    if result_op is None:
        result_prec, result_assoc = precedence, assoc
    else:
        result_prec, result_assoc = Precedence_lookup(result_op)
    return (left_expr[0] + right_expr[0],
            format.format(wrap(left_expr, precedence, 'left'),
                          wrap(right_expr, precedence, 'right')),
            result_precedence, result_assoc)


expr_unary = {
    'NOT': partial(unary, '!', '!{}'),
    '-': partial(unary, 'UMINUS', '-{}'),
}


def subscript(left, right):
    precedence, assoc = Precedence_lookup['[']
    return (f"subscript({wrap(left, precedence, assoc)}, {right[0]}, "
              f"{left.dim()})",
            precedence, assoc)


expr_binary = {
    '.': partial(binary, '.', '{}.{}'),
    '^': partial(binary, ',', 'pow({}, {})', result_op='('),
    '*': partial(binary, '*', '{} * {}'),
    '/': partial(binary, '/', '{} / {}'),
    '%': partial(binary, '%', '{} % {}'),
    '+': partial(binary, '+', '{} + {}'),
    '-': partial(binary, '-', '{} - {}'),
    '<': partial(binary, '<', '{} < {}'),
    'LEQ': partial(binary, '<=', '{} <= {}'),
    'LAEQ': partial(binary, ',', 'laeq({}, {})', result_op='('),
    '>': partial(binary, '>', '{} > {}'),
    'GEQ': partial(binary, '>=', '{} >= {}'),
    'GAEQ': partial(binary, ',', 'gaeq({}, {})', result_op='('),
    'EQ': partial(binary, '==', '{} == {}'),
    'AEQ': partial(binary, ',', 'aeq({}, {})', result_op='('),
    'NEQ': partial(binary, '!=', '{} != {}'),
    'NAEQ': partial(binary, ',', 'naeq({}, {})', result_op='('),
    '[': subscript,
}


types = {
    "float": "double",
    "integer": "int",
    "bool": "bool",
    "string": "char *",
}


preamble = '''

struct sub_ret_s {
    void *module_context;
    void *label;
};

struct pos_param_block__f {
    double param1_f;
    double param2_f;
    double param3_f;
    double param4_f;
    double param5_f;
    double param6_f;
};

struct fn_ret_s {
    struct sub_ret_s sub_ret_info;
    struct pos_param_block__f *ret_params;
};

struct pos_param_bloc__f_with_return {
    struct sub_ret_s sub_ret_info;
    struct pos_param_block__f *params;
};

'''

def gen_program(opmode):
    print(f"// {opmode.name.value}.c")
    print(preamble)

def translate_type(type):
    return types.get(type.lower(), type)

def emit_struct_start(name):
    global struct_temps, struct_seen_dlt_mask
    print()
    print(indent_str(Struct_indent), f"struct {name} {{", sep='')
    struct_temps = []
    struct_seen_dlt_mask = False
    return f"struct {name}"

def emit_struct_variable(type, name, dim=None):
    _emit_struct_variable(types[type], name, dim)

def _emit_struct_variable(type, name, dim=None):
    if dim is None:
        print(indent_str(Struct_var_indent), f"{type} {name};", sep='')
    else:
        print(indent_str(Struct_var_indent), f"{type} {name}[{dim}];",
              sep='')

def emit_struct_sub_ret_info():
    _emit_struct_variable("struct sub_ret_s", "sub_ret_info")

def emit_struct_fn_ret_info():
    _emit_struct_variable("struct fn_ret_s", "fn_ret_info")

def emit_struct_dlt_mask():
    global struct_seen_dlt_mask
    struct_seen_dlt_mask = True

def emit_struct_end():
    for type, name in struct_temps:
        emit_struct_variable(type, name)
    if struct_seen_dlt_mask:
        _emit_struct_variable('unsigned long', 'dlt_mask')
    print(indent_str(Struct_indent), "};", sep='')

def emit_code_start():
    print("int main(int argc, char **argv, char **env) {")
    print(indent_str(Label_indent), "void *current_module;", sep='')
    print()

def emit_code_end():
    print("}")

def emit_label(name):
    print(indent_str(Label_indent), f"{name}:", sep='')

def emit_statements(statements):
    for line in lines:
        print(indent_str(Statement_indent), line, sep='')

def reset_temps():
    r'''Called at the start of each statement.gen_code
    '''
    pass

def gen_expr(expr):
    r'''Returns (statements,), C code, C precedence level, C associativity
    '''
    if isinstance(expr, str):
        C_str = expr.replace('""', '\"')
        return (), f'"{C_str}"', 100, 'nonassoc'
    if isinstance(expr, (float, int)):
        return (), repr(expr), 100, 'nonassoc'
    if isinstance(expr, bool):
        return (), ("1" if expr else "0"), 100, 'nonassoc'
    if isinstance(expr, (tuple, list)):
        if len(expr) == 2:
            return convert_unary(expr[0], gen_expr(expr[1]))
        elif expr[0] == 'call_fn':
            return convert_call_fn(gen_expr(expr[1]), expr[2])
        else:
            return convert_binary(gen_expr(expr[0]), expr[1], gen_expr(expr[2]))
    return convert_token(expr)

def gen_lvalue(lvalue):
    r'''Returns (statements,), C code, C precedence level, C associativity

    lvalue : IDENT
           | primary '.' IDENT
           | primary '[' expr ']'
    '''
    return gen_expr(lvalue)

def convert_token(token):
    # FIX
    print("convert_atom", token)
    return (), token.value, 100, 'nonassoc'

def convert_unary(oper, expr):
    print("convert_unary", oper, expr)
    return expr_unary[oper](oper, expr)

def convert_binary(left_expr, oper, right_expr):
    print("convert_binary", left_expr, oper, right_expr)
    return expr_binary[oper](left_expr, oper, right_expr)

def convert_call_fn(fn, arguments):
    # FIX
    print("convert_call_fn", fn, arguments)
    return (), "call_fn", 100, 'nonassoc'
