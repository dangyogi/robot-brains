# C_gen.py

import sys
import os.path
import operator
from functools import partial

import symtable
from scanner import syntax_error


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
    "bool": "char",
    "string": "char *",
}


def gen_program(opmode):
    global C_file

    c_filename = f"{opmode.name.value}.c"

    with open(c_filename, 'wt') as C_file:

        # This is modules, _not_ uses (instances)!
        # ... i.e., each of these may have multiple instances within the
        #           program.
        # These are in bottom-up order based on "uses".
        modules = list(tsort_modules(opmode))
        #print("modules", modules)


        def do(fn):
            ans = []
            for m in modules:
                result = fn(m)
                if result:
                    ans.extend(result)
            return ans

        print("prepare ....")
        do(operator.methodcaller('prepare', opmode))
        print("assign_names ....")
        do(assign_names)
        print("gen_code ....")
        lines = do(gen_code)

        print(f"// {opmode.name.value}.c", file=C_file)

        # Copy in C_preamble file
        with open("C_preamble") as preamble:
            for line in preamble:
                C_file.write(line)

        print("gen_typedefs ....")
        do(gen_typedefs)
        print("gen_descriptors ....")
        do(gen_descriptors)

        print("gen_globals ....")
        do(gen_globals)
        print("emit_code_start ....")
        emit_code_start()
        print("gen_start_labels ....")
        do(gen_start_labels)
        print("gen_init ....")
        do(gen_init)
        for line in lines:
            print(line, file=C_file)
        print("emit_code_end ....")
        emit_code_end()


def assign_names(m):
    if isinstance(m, symtable.Opmode):
        m.C_global_name = translate_name(m.name.value)
    else:
        m.C_global_name = \
          "__".join((os.path.basename(os.path.dirname(m.filename)),
                     translate_name(m.name.value)))
    assign_child_names(m)


def assign_child_names(parent):
    for obj in parent.names.values():
        if isinstance(obj, symtable.Variable):
            # name defined inside namespace struct
            if obj.assignment_seen:
                obj.C_local_name = translate_name(obj.name.value)
        elif isinstance(obj, symtable.Use):
            # name defined inside namespace struct
            obj.C_local_name = translate_name(obj.name.value)
        elif isinstance(obj, symtable.Typedef):
            # name defined globally
            obj.C_global_name = \
              parent.C_global_name + "__" + translate_name(obj.name.value)
        elif isinstance(obj, symtable.Labeled_block):
            # name defined globally
            obj.C_global_name = \
              parent.C_global_name + "__" + translate_name(obj.name.value)
        elif isinstance(obj, symtable.Subroutine):
            obj.C_local_name = translate_name(obj.name.value)
            obj.C_global_name = parent.C_global_name + "__" + obj.C_local_name
            obj.C_temps = []  # [(type, name)]
            assign_child_names(obj)
        else:
            # This seems to only be References
            if (False):
                print("assign_child_names: ignoring",
                      obj.__class__.__name__, obj.name.value, "in",
                      parent.__class__.__name__, parent.name.value)


def translate_name(name):
    if not name[-1].isdecimal() and not name[-1].isalpha():
        return name[:-1].lower() + '_'
    return name.lower()


def translate_type(type):
    return types.get(type.lower(), type)

ret_struct_names = {
    'FLOAT': 'fn_ret_f_s',
    'INTEGER': 'fn_ret_i_s',
    'STRING': 'fn_ret_s_s',
    'BOOL': 'fn_ret_b_s',
}


def gen_typedefs(module):
    for obj in module.names.values():
        if isinstance(obj, symtable.Typedef):
            if obj.definition[0].value.upper() == 'FUNCTION':
                if obj.definition[2]:
                    # explicit RETURNING clause
                    if len(obj.definition[2]) <= 6 \
                       and len(frozenset(r.value.upper()
                                         for r in obj.definition[2])) == 1 \
                       and obj.definition[0].value.upper() in ret_struct_names:
                        # <= 6 return parameters, all same fundamental type
                        ret_struct = \
                          ret_struct_names[obj.definition[0].value.upper()]
                    else:
                        print("gen_typedefs: creating new return type for",
                              obj.name.value, "in", module.name.value,
                              file=sys.stderr)
                        ret_struct = f"{obj.C_global_name}__ret_s"
                        emit_struct_start(ret_struct)
                        for i, t in enumerate(obj.definition[2], 1):
                            emit_struct_variable(t, f"param_{i}")
                        emit_struct_end()
                else:
                    # no explicit RETURNING clause, take return type from name
                    ret_struct = ret_struct_names[obj.name.type.upper()]
                ret_name = 'fn_ret_info'
            else: # SUBROUTINE
                ret_struct = 'sub_ret_s'
                ret_name = 'sub_ret_info'
            emit_struct_start(obj.C_global_name)
            _emit_struct_variable(f"struct {ret_struct}", ret_name)
            for i, t in enumerate(obj.definition[1], 1):
                emit_struct_variable(t, f"param_{i}")
            emit_struct_end()
        elif isinstance(obj, symtable.Subroutine):
            for child in obj.names.values():
                gen_typedefs(child)


def gen_code(m):
    lines = []
    for sub in m.names.values():
        if isinstance(sub, symtable.Subroutine):

            def gen_label(label):
                lines.append('')
                lines.append(prepare_label(label))
                return label

            def gen_init_defaults(parameters, num_defaults, label_format):
                # generate start labels and default code:
                num_req_params = len(parameters) - num_defaults
                labels = ['too_few_arguments' * num_req_params]
                for i in range(num_req_params, len(parameters)):
                    labels.append(gen_label(label_format.format(i)))
                    # gen default code
                    statements, C_expr, _, _ = gen_expr(parameters[i][1])
                    lines.extend(prepare_statements(statements))
                    lvalue = gen_lvalue(parameters[i][0])[1]
                    lines.extend(prepare_statements([f"{lvalue} = {C_expr};"]))
                return labels

            # generate start labels and pos defaults:
            label_format = f"{sub.C_global_name}_start_{{}}"
            sub.start_labels = \
              gen_init_defaults(sub.pos_parameters, sub.num_pos_defaults,
                                label_format)
            if sub.no_kw_defaults:
                # This final label will fall-through to the subroutine code
                sub.start_labels.append(
                  gen_label(label_format.format(len(sub.pos_parameters))))
            else:
                need_return = bool(sub.num_pos_defaults)
                # generate keyword defaults.  The last of these will
                # fall-through to the subroutine code.
                sub.kw_labels = {}  # {kw: [labels]}
                                    # excludes labels[num_params] --
                                    #    no defaults left to initialize!
                for kw, params in self.kw_parameters.items():
                    if need_return:
                        gen_sub_return(sub)
                    sub.kw_labels[kw] = gen_init_defaults(
                      params,
                      sub.num_kw_defaults[kw],
                      f"{sub.C_global_name}_kw_param_init_{{}}")
                    need_return = bool(sub.num_kw_defaults[kw])

            if sub.first_block:
                lines.extend(gen_block(sub.first_block))
            for labeled_block in sub.names.values():
                if isinstance(labeled_block, symtable.Labeled_block):
                    if labeled_block.seen_default:
                        syntax_error("default parameters on labels not yet "
                                       "implemented",
                                     labeled_block.name.lexpos,
                                     labeled_block.name.lineno,
                                     labeled_block.name.filename,
                                     labeled_block.name.namespace)
                    for param in labeled_block.pos_parameters:
                        # FIX
                        pass
                    lines.extend(gen_block(labeled_block.block))
    return lines


def gen_block(block):
    lines = []
    for s in block.statements:
        # FIX
        pass
    return lines


def gen_globals(m):
    structs = []
    names = []
    for obj in m.names.values():
        if isinstance(obj, symtable.Subroutine):
            structs.append(obj.C_global_name)
            names.append(obj.C_local_name)
            emit_struct_start(obj.C_global_name)
            if isinstance(obj, symtable.Function):
                _emit_struct_variable(
                  f'struct {ret_struct_names[obj.name.type]}',
                  'fn_ret_info')
            else:
                _emit_struct_variable('struct sub_ret_s', 'sub_ret_info')
            for var in obj.names.values():
                if isinstance(var, symtable.Variable) and var.assignment_seen:
                    emit_struct_variable(var.type, var.C_local_name)
            if obj.dlt_mask_needed:
                _emit_struct_variable('unsigned long', 'dlt_mask')
            for t, n in obj.C_temps:
                _emit_struct_variable(t, n)
            emit_struct_end()
    emit_struct_start(translate_name(m.name.value))
    for s, n in zip(structs, names):
        _emit_struct_variable(f"struct {s}", n)
    emit_struct_end()


def emit_struct_start(name):
    global struct_temps, struct_seen_dlt_mask
    print(file=C_file)
    print(symtable.indent_str(Struct_indent), f"struct {name} {{",
          sep='', file=C_file)
    struct_temps = []
    struct_seen_dlt_mask = False
    return f"struct {name}"


def emit_struct_variable(type, name, dim=None):
    _emit_struct_variable(types[type], name, dim)


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
    print(symtable.indent_str(Struct_indent), "};", sep='', file=C_file)


def emit_code_start():
    print(file=C_file)
    print("int main(int argc, char **argv, char **env) {", file=C_file)
    print(symtable.indent_str(Label_indent), "void *current_module;",
          sep='', file=C_file)
    print(file=C_file)


def emit_code_end():
    print("}", file=C_file)


def prepare_label(name):
    return f"{symtable.indent_str(Label_indent)}{name}:"


def prepare_statements(statements):
    return [f"{symtable.indent_str(Statement_indent)}{line}"
            for line in statements]


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


def tsort_modules(opmode):
    r'''Generates modules in bottom-up order according to "uses".

    opmode.modules_seen is {name: module}.
    '''
    for name, m in opmode.modules_seen.items():
        assert name == m.name.value

    # {parent: set(children)}
    order = dict(gen_dependencies(opmode.modules_seen))
    #print("tsort_modules, order:", order)
    for children in order.values():
        for child in children:
            assert child in order

    while order:
        leaves = frozenset(parent
                           for parent, children in order.items()
                            if not children)
        for module in sorted(leaves, key=lambda m: m.name.value):
            yield module
            del order[module]
        for children in order.values():
            children -= leaves
    yield opmode


def gen_dependencies(modules_seen):
    for m in modules_seen.values():
        yield (m,
               set(modules_seen[x.module_name.value]
                   for x in m.names.values()
                    if isinstance(x, symtable.Use)))
