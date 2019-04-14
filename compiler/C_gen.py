# C_gen.py

import sys
import os.path
import operator
from functools import partial

import symtable
from scanner import syntax_error

# 32-bit machine:
#Sizeof_routine_instance_s = 24

# 64-bit machine:
Sizeof_routine_instance_s = 48
Sizeof_long = 8
Sizeof_double = 8
Sizeof_ptr = 8
Sizeof_bool = 1
Align_long = 8
Align_double = 8
Align_ptr = 8
Align_bool = 1


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

        print(file=C_file)
        print(file=C_file)
        print("// structs", file=C_file)
        print("gen_structs ....")
        do(gen_structs)

        print("gen_descriptors ....")
        print(file=C_file)
        print(file=C_file)
        print("// descriptors", file=C_file)
        do(gen_descriptors)

        print(file=C_file)
        print(file=C_file)
        print("// instances", file=C_file)
        print("gen opmode instance ....")
        gen_instance(opmode)
        print("gen builtins instance ....")
        gen_instance(opmode.modules_seen['builtins'])

        print("emit_code_start ....")
        print(file=C_file)
        print(file=C_file)
        print("// code start", file=C_file)
        emit_code_start()

        print(file=C_file)
        print(file=C_file)
        print("// call init routines", file=C_file)
        emit_subroutine_call('builtins__module__init')
        emit_subroutine_call(f'{opmode.name.value}__opmode__init')

        print("write code ....")
        print(file=C_file)
        print(file=C_file)
        print("// generated code", file=C_file)
        for line in lines:
            print(line, file=C_file)

        print("emit_code_end ....")
        emit_code_end()


def assign_names(m):
    if isinstance(m, symtable.Module):
        m.C_global_name = \
          "__".join((os.path.basename(os.path.dirname(m.filename)),
                     translate_name(m.name.value))) + "__module"
    else:
        m.C_global_name = translate_name(m.name.value) + "__opmode"
    m.C_descriptor_name = m.C_global_name + "__desc"
    m.C_struct_name = m.C_global_name + "__instance_s"
    m.C_init_subroutine_name = m.C_global_name + "__init__"
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
            obj.C_start_label_name = obj.C_global_name + "__start"
            obj.C_struct_name = obj.C_global_name + "__instance_s"
            obj.C_descriptor_name = obj.C_global_name + "__desc"
            obj.C_temps = []  # [(type, name)]
            obj.needs_dlt_mask = False
            assign_child_names(obj)
        else:
            # This seems to only be References
            if (False):
                print("assign_child_names: ignoring",
                      obj.__class__.__name__, obj.name.value, "in",
                      parent.__class__.__name__, parent.name.value)

    if not isinstance(parent, symtable.Opmode):
        if parent.pos_param_block is not None:
            assign_param_names(parent, parent.pos_param_block)
        for pb in parent.kw_parameters.values():
            assign_param_names(parent, pb)


def assign_param_names(parent, param_block):
    if param_block.optional_params:
        param_block.optional_param_block_struct_name = \
          parent.C_global_name + "__" + param_block.name
        param_block.optional_param_block_name = \
          parent.C_local_name + "__" + param_block.name + "__optionals"
        for p in param_block.optional_params:
            p.C_local_name = \
              f"{param_block.optional_param_block_name}.{p.C_local_name}"


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


def gen_descriptors(module):
    routines = tuple(get_routines(module))
    for r in routines:
        gen_routine_descriptor(r)

    # generate module descriptor
    print(file=C_file)
    print(f"struct module_descriptor_s {module.C_global_name} = {{",
          file=C_file)
    print(f'  "{module.name.value}",', file=C_file)
    print(f'  "{module.filename}",', file=C_file)
    uses = tuple(get_uses(module))
    print(f'  {len(uses)}, // num_uses', file=C_file)
    print(f'  {len(routines)}, // num_routines', file=C_file)
    print('  (struct use_descriptor_s []){  // uses', file=C_file)
    for u in uses:
        print(f'    {{"{u.name.value}", 0}},', file=C_file)
    print("  },", file=C_file)
    print('  {  // routines', file=C_file)
    for r in routines:
        print(f'    &{r.C_global_name},', file=C_file)
    print("  },", file=C_file)
    print("};", file=C_file)


def gen_routine_descriptor(routine):
    print(file=C_file)
    print(f"struct routine_descriptor_s {routine.C_global_name} = {{",
          file=C_file)
    print(f'  "{routine.name.value}",', file=C_file)
    type = 'F' if isinstance(routine, symtable.Function) else 'S'
    print(f"  '{type}',", file=C_file)
    if routine.pos_param_block is None:
        num_param_blocks = 0
    else:
        num_param_blocks = 1
    print(f'  {num_param_blocks},  // num_param_blocks', file=C_file)
    print('  (struct param_block_descriptor_s []){  // param_block_descriptors',
          file=C_file)
    if routine.pos_param_block is not None:
        gen_param_desc(routine.pos_param_block)
    for kw_params in routine.kw_parameters.values():
        gen_param_desc(kw_params)
    print("  },", file=C_file)
    print("};", file=C_file)


def gen_param_desc(param_block):
    print('    {  // param_block_descriptor', file=C_file)
    print(f'      "{param_block.name}",  // name', file=C_file)
    num_params = len(param_block.required_params) \
               + len(param_block.optional_params)
    print(f'      {num_params},  // num_params', file=C_file)
    print(f'      {len(param_block.required_params)},  // num_required_params',
          file=C_file)
    offsets = []
    # FIX add required_param offsets
    if not param_block.optional_params:
        print(f'      0,  // defaults_size', file=C_file)
        print(f'      NULL,  // defaults_block', file=C_file)
    else:
        print(f'      sizeof(struct {param_block.optional_param_block_struct_name})'
              ',  // defaults_size',
              file=C_file)
        default_block = []
        current_offset = Sizeof_routine_instance_s
        def append(t):
            # FIX
            pass
        for p in default_params:
            append(p)
        print(f'      &(struct {param_block.optional_param_block_struct_name})'
              f'{{{", ".join(default_block)}}},  // defaults_block',
              file=C_file)
    print(f'      (int []){{{", ".join(str(i) for i in offsets)}}},'
          '  // param_offsets',
          file=C_file)
    print('    },', file=C_file)


def get_uses(module):
    for v in module.names.values():
        if isinstance(v, symtable.Use):
            yield v


def get_routines(module):
    for v in module.names.values():
        if isinstance(v, symtable.Subroutine):
            yield v


def gen_structs(module):
    for sub in get_routines(module):
        gen_routine_struct(sub)

    emit_struct_start(module.C_struct_name)
    _emit_struct_variable("struct module_descriptor_s *", 'descriptor')
    for obj in module.names.values():
        if isinstance(obj, symtable.Use):
            _emit_struct_variable(f"struct {obj.module.C_struct_name} ",
                                  obj.C_local_name)
        elif isinstance(obj, symtable.Variable):
            emit_struct_variable(obj.type, obj.C_local_name, obj.dim)
        elif isinstance(obj, symtable.Subroutine):
            _emit_struct_variable(f"struct {obj.C_struct_name}",
                                  obj.C_local_name)
    emit_struct_end()


def gen_routine_struct(routine):
    emit_struct_start(routine.C_struct_name)
    _emit_struct_variable("struct routine_instance_s", '__base__')
    emit_param_blocks(routine)
    for obj in routine.names.values():
        if isinstance(obj, symtable.Required_parameter):
            # skip, handled by emit_param_blocks...
            # FIX, we need to allow multiple labels to declare the same params!
            pass
        elif isinstance(obj, symtable.Variable):
            emit_struct_variable(obj.type, obj.C_local_name, obj.dim)
        elif isinstance(obj, symtable.Labeled_block):
            emit_param_blocks(obj)
    for type, name in routine.C_temps:
        emit_struct_variable(type, name)
    if routine.needs_dlt_mask:
        _emit_struct_variable('unsigned long', 'dlt_mask')
    emit_struct_end()


def emit_param_blocks(obj):
    if obj.pos_param_block is not None:
        emit_param_block(obj.pos_param_block)
    for pb in obj.kw_parameters.values():
        emit_param_block(pb)


def emit_param_block(pb):
    for p in pb.required_params:
        emit_struct_variable(p.type, p.C_local_name, p.dim)
    if pb.optional_params:
        _emit_struct_variable(
          f"struct {pb.optional_param_block_struct_name }", 
          pb.optional_param_block_name)


def gen_instance(module):
    # FIX
    pass


def gen_code(m):
    lines = []

    def gen_label(label):
        lines.append('')
        lines.append(prepare_label(label))
        return label

    def gen_statement(line):
        lines.append(prepare_statement(line))

    def gen_call(label):
        # FIX: complete this...
        gen_statement(f"goto {label};")

    gen_label(m.C_init_subroutine_name)

    for use in get_uses(m):
        gen_call(use.module.C_init_subroutine_name)

    for sub in get_routines(m):
        gen_statement(
          f'{sub.C_local_name}.__base__.descriptor = &{sub.C_descriptor_name};')
        gen_statement(f'{sub.C_local_name}.__base__.flags = 0;')
        gen_statement(
          f'{sub.C_local_name}.__base__.global = &module_instance;')

    for sub in get_routines(m):
        gen_label(sub.C_start_label_name)

        # generate start labels and pos defaults:

        if sub.first_block:
            lines.extend(gen_block(sub.first_block))
        for labeled_block in sub.names.values():
            if isinstance(labeled_block, symtable.Labeled_block):
                gen_label(labeled_block.C_global_name)
                lines.extend(gen_block(labeled_block.block))

    return lines


def gen_block(block):
    lines = []
    for s in block.statements:
        lines.append(prepare_statement(
                       f"current_routine->lineno = {s.lineno};"))
        # FIX
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
            #if obj. ....dlt_mask_needed:
            #    _emit_struct_variable('unsigned long', 'dlt_mask')
            for t, n in obj.C_temps:
                _emit_struct_variable(t, n)
            emit_struct_end()
    emit_struct_start(translate_name(m.name.value))
    for s, n in zip(structs, names):
        _emit_struct_variable(f"struct {s}", n)
    emit_struct_end()


def emit_struct_start(name):
    print(file=C_file)
    print(symtable.indent_str(Struct_indent), f"struct {name} {{",
          sep='', file=C_file)


def emit_struct_variable(type, name, dim=None):
    _emit_struct_variable(types[type], name, dim)


def _emit_struct_variable(type, name, dim=None):
    print(symtable.indent_str(Struct_var_indent), type,
          sep='', end='', file=C_file)
    if not type.endswith('*'):
        print(' ', end='', file=C_file)
    print(f"{name};", file=C_file)


def emit_struct_end():
    print(symtable.indent_str(Struct_indent), "};", sep='', file=C_file)


def emit_code_start():
    print(file=C_file)
    print(file=C_file)
    print("int", file=C_file)
    print("main(int argc, char **argv, char **env) {", file=C_file)
    print(symtable.indent_str(Label_indent),
          "struct routine_instance_s *current_routine;",
          sep='', file=C_file)


def emit_code_end():
    print("}", file=C_file)


def emit_subroutine_call(label_name):
    # FIX
    pass


def prepare_label(name):
    return f"{symtable.indent_str(Label_indent)}{name}:"


def prepare_statements(statements):
    return [prepare_statement(line) for line in statements]


def prepare_statement(line):
    return f"{symtable.indent_str(Statement_indent)}{line}"


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
