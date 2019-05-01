# C_gen.py

import sys
import os.path
import operator
from functools import partial

import symtable
from scanner import Token, syntax_error


C_UNSIGNED_LONG_BITS = 64
Num_label_bits = 1              # running bit
Max_bits_used = 0


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


C_reserved_words = frozenset((
    'auto',
    'break',
    'case',
    'char',
    'const',
    'continue',
    'default',
    'do',
    'double',
    'else',
    'enum',
    'extern',
    'float',
    'for',
    'goto',
    'if',
    'int',
    'long',
    'register',
    'return',
    'short',
    'signed',
    'sizeof',
    'static',
    'struct',
    'switch',
    'typedef',
    'union',
    'unsigned',
    'void',
    'volatile',
    'while',
))


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
    "boolean": "char",
    "string": "char *",
    "module": "struct module_instance_s",
}


def gen_program(opmode):
    global C_file

    c_filename = f"{opmode.name}.c"

    with open(c_filename, 'wt') as C_file:

        # This is modules, _not_ uses (instances)!
        # ... i.e., each of these may have multiple instances within the
        #           program.
        # These are in bottom-up order based on "uses".
        modules = list(tsort_modules(opmode))
        #print("modules", modules)

        def do(fn, top_down=False):
            ans = []
            for m in (reversed(modules) if top_down else modules):
                result = fn(m)
                if result:
                    ans.extend(result)
            return ans

        print("prepare ....")
        do(lambda m: m.prepare(opmode, m), top_down=True)
        print("assign_names ....")
        do(assign_names)
        print("Max_bits_used", Max_bits_used, Max_bits_obj)
        if Max_bits_used >= C_UNSIGNED_LONG_BITS:
            print("Exceeded number of available label flags bits in",
                  Max_bits_obj.name.value,
                  file=sys.stderr)
            sys.exit(1)
        print("gen_code ....")
        lines = do(gen_code)

        print(f"// {opmode.name}.c", file=C_file)

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
        print("gen opmode and builtin module instances ....")
        gen_instance(opmode)
        print("gen builtins instance ....")
        builtins_module = opmode.modules_seen['builtins']
        gen_instance(builtins_module)

        print(file=C_file)
        print(file=C_file)
        print("// modules_list", file=C_file)
        print("modules_list ....")
        instances = tuple(gen_instances(builtins_module)) \
                  + tuple(gen_instances(opmode))
        print(f"const int num_module_instances = {len(instances)};",
              file=C_file)
        print("const struct module_instance_s module_instances = {",
              " // bottom-up order",
              file=C_file)
        for m in instances:
            print(f"  (struct module_instance_s *)&{m},", file=C_file)
        print("};", file=C_file)

        print("emit_code_start ....")
        print(file=C_file)
        print(file=C_file)
        print("// code start", file=C_file)
        emit_code_start()

        print("init descriptors ....")
        print(file=C_file)
        print(file=C_file)
        print("// init descriptors", file=C_file)
        do(init_descriptors)

        # FIX XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        print("call init subroutines ....")
        print(file=C_file)
        print(file=C_file)
        print(prepare_statement("// call init routines"), file=C_file)
        print(file=C_file)
        print(prepare_statement(f"{builtins_module.C_init_C_fn_name}("
                                f"&{builtins_module.C_global_name});"),
              file=C_file)
        print(file=C_file)
        print(prepare_statement(f"{opmode.C_init_C_fn_name}("
                                f"&{opmode.C_global_name});"),
              file=C_file)

        print("call cycle.run ....")
        print(file=C_file)
        print(file=C_file)
        print(prepare_statement("// call cycle run"), file=C_file)
        print(file=C_file)

        # find "cycle" module (look in opmode and builtins):
        for use in get_uses(opmode):
            if use.name.value.lower() == 'cycle':
                cycle_module = use.module
                cycle_module_name = f'{opmode.C_global_name}.{use.C_local_name}'
                break
        else:
            for use in get_uses(builtins_module):
                if use.name.value.lower() == 'cycle':
                    cycle_module = use.module
                    cycle_module_name = \
                      f'{builtins_module.C_global_name}.{use.C_local_name}'
                    break
            else:
                print("could not find cycle module for cycle.run call",
                      file=sys.stderr)
                sys.exit(1)

        # find "run" subroutine:
        for label in get_labels(cycle_module):
            if label.name.value.lower() == 'run':
                if not isinstance(label, symtable.Subroutine) \
                   or isinstance(label, symtable.Function):
                    print('cycle.run must be a subroutine, '
                            f'not a {label.__class__.__name__}',
                          file=sys.stderr)
                    sys.exit(1)
                if label.pos_param_block is not None or label.kw_parameters:
                    print('cycle.run must not take any parameters',
                          file=sys.stderr)
                    sys.exit(1)
                run_label = label
                run_label_name = f"{cycle_module_name}.{label.C_local_name}"
                break
        else:
            print('cycle module does not have a "run" subroutine',
                  file=sys.stderr)
            sys.exit(1)

        print(prepare_statement(
                f"{run_routine_name}.__base__.ret_pointer.module = "
                "current_module;"),
              file=C_file)
        print(prepare_statement(
                f"{run_routine_name}.__base__.ret_pointer.descriptor = FIX;"),
              file=C_file)
        print(prepare_statement(
                "current_module = "
                f"(struct module_instance_s *)&{run_routine_name};"),
              file=C_file)
        print(prepare_statement(
                f"goto {run_routine.C_start_label_name};"),
              file=C_file)

        print("terminate_program_0 ....")
        print(file=C_file)
        print(file=C_file)
        print(prepare_statement("// all done!"), file=C_file)
        print(file=C_file)
        print(prepare_label("terminate_program_0"), file=C_file)
        print(prepare_statement("return 0;"), file=C_file)

        print("write code ....")
        print(file=C_file)
        print(file=C_file)
        print(prepare_statement("// generated code"), file=C_file)
        for line in lines:
            print(line, file=C_file)

        print("gen_init_fns ....")
        print(file=C_file)
        print(file=C_file)
        print(prepare_statement("// init_fns"), file=C_file)
        do(gen_init_fns)

        print("emit_code_end ....")
        emit_code_end()


def assign_names(m):
    # set C_global_name: globally unique name
    if isinstance(m, symtable.Module):
        m.C_global_name = \
          "__".join((os.path.basename(os.path.dirname(m.filename)),
                     translate_name(m.name))) + "__module"
    else:
        m.C_global_name = translate_name(m.name) + "__opmode"
    m.C_descriptor_name = m.C_global_name + "__desc"
    m.C_struct_name = m.C_global_name + "__instance_s"
    m.C_init_sub_name = m.C_global_name + "__init__"
    assign_child_names(m)


def assign_child_names(module):
    for obj in module.names.values():
        if isinstance(obj, symtable.Variable):
            # name defined inside namespace struct
            obj.C_type = translate_type(obj.type)
            obj.C_local_name = translate_name(obj.name.value)
        elif isinstance(obj, symtable.Use):
            # name defined inside namespace struct
            obj.C_local_name = translate_name(obj.name.value)
        elif isinstance(obj, symtable.Typedef):
            obj.C_type = translate_type(obj.type)
            # name defined globally
            obj.C_global_name = \
              module.C_global_name + "__" + translate_name(obj.name.value)
        elif isinstance(obj, symtable.Labeled_block):
            # unique inside module
            obj.C_local_name = translate_name(obj.name.value)

            obj.C_flags_name = obj.C_local_name + "__flags__"
            if isinstance(obj, symtable.Subroutine):
                obj.C_ret_pointer_name = obj.C_local_name + "__ret_pointer__"

            # name defined globally
            obj.C_global_name = module.C_global_name + "__" + obj.C_local_name

            obj.C_descriptor_name = obj.C_global_name + "__desc__"
            obj.C_label_name = obj.C_global_name + "__label__"
            obj.C_dlt_mask_name = obj.C_local_name + "__dlt_mask__"
            obj.needs_dlt_mask = False
            obj.C_temps = []  # [(type, name)]
            assign_parameters(obj)
        else:
            # This seems to only be References
            if (True):
                print("assign_child_names: ignoring",
                      obj.__class__.__name__, obj.name.value, "in",
                      module.__class__.__name__, module.name.value)

    if not isinstance(module, symtable.Opmode):
        assign_parameters(module)


def assign_parameters(obj):
    global Max_bits_used, Max_bits_obj
    for pb in obj.gen_param_blocks():
        assign_param_names(obj, pb)
    if obj.bits_used + 2 > Max_bits_used:
        Max_bits_used = obj.bits_used + 2
        Max_bits_obj = obj


def assign_param_names(parent, param_block):
    if param_block.optional:
        param_block.C_kw_passed_mask = 1 << param_block.kw_passed_bit
    else:
        param_block.C_kw_passed_mask = 0
    for p in param_block.optional_params:
        p.C_param_mask = 1 << p.passed_bit


def translate_name(name):
    if not name[-1].isdecimal() and not name[-1].isalpha():
        return name[:-1].lower() + '_'
    if name.lower() in C_reserved_words:
        return name.lower() + '_'
    return name.lower()


def translate_type(type):
    if isinstance(type, symtable.Builtin_type):
        return types[type.name.lower()]
    if isinstance(type, symtable.Label_type):
        return "struct label_pointer_s"
    assert isinstance(type, symtable.Typename_type)
    return translate_type(type.typedef.type)


ret_struct_names = {
    'FLOAT': 'fn_ret_f_s',
    'INTEGER': 'fn_ret_i_s',
    'STRING': 'fn_ret_s_s',
    'BOOL': 'fn_ret_b_s',
}


def gen_descriptors(module):
    labels = tuple(get_labels(module))
    for label in labels:
        gen_label_descriptor(module, label)

    # generate module descriptor
    print(file=C_file)
    print(f"struct module_descriptor_s {module.C_descriptor_name} = {{",
          file=C_file)
    print(f'  "{module.name}",  // name', file=C_file)
    print(f'  "{module.filename}",  // filename', file=C_file)
    print(f'  {len(labels)},  // num_labels', file=C_file)
    print('  {  // labels', file=C_file)
    for label in labels:
        print(f'    &{label.C_descriptor_name},', file=C_file)
    print("  },", file=C_file)
    print("};", file=C_file)


def gen_label_descriptor(module, label):
    print(file=C_file)
    print(f"struct label_descriptor_s {label.C_descriptor_name} = {{",
          file=C_file)
    print(f'  "{label.name.value}",  // name', file=C_file)
    if isinstance(label, symtable.Function):
        type = 'F'
    elif isinstance(label, symtable.Subroutine):
        type = 'S'
    else:
        assert isinstance(label, symtable.Labeled_block)
        type = 'L'
    print(f"  '{type}',   // type", file=C_file)
    if label.pos_param_block is None:
        num_param_blocks = 0
    else:
        num_param_blocks = 1
    num_param_blocks += len(label.kw_parameters)
    print(f'  {num_param_blocks},  // num_param_blocks', file=C_file)
    print('  (struct param_block_descriptor_s []){  // param_block_descriptors',
          file=C_file)
    for pb in label.gen_param_blocks():
        gen_param_desc(module, pb)
    print("  },  // param_block_descriptors", file=C_file)
    print(f'  offsetof(struct {module.C_struct_name}, {label.C_flags_name}),'
            '  // flags_offset',
          file=C_file)
    if isinstance(label, symtable.Subroutine):
        print(f'  offsetof(struct {module.C_struct_name}, {label.C_ret_pointer_name}),'
                '  // ret_pointer_offset',
              file=C_file)
    else:
        print('  0,  // ret_pointer_offset', file=C_file)
    print("};", file=C_file)


def gen_param_desc(module, param_block):
    print('    {  // param_block_descriptor', file=C_file)
    print(f'      "{param_block.name}",  // name', file=C_file)
    num_params = len(param_block.required_params) \
               + len(param_block.optional_params)
    print(f'      {num_params},  // num_params', file=C_file)
    print(f'      {len(param_block.required_params)},  // num_required_params',
          file=C_file)
    print(f'      {len(param_block.optional_params)},  // num_optional_params',
          file=C_file)

    offsets = []
    param_masks = []

    # add offsets to required_params
    for p in param_block.required_params:
        offsets.append(f"offsetof(struct {module.C_struct_name}, "
                       f"{p.variable.C_local_name})")

    # add offsets and param_masks for optional_params
    for p in param_block.optional_params:
        offsets.append(f"offsetof(struct {module.C_struct_name}, "
                       f"{p.variable.C_local_name})")
        param_masks.append(p.passed_bit)

    print('      (int []){', ', '.join(str(offset) for offset in offsets),
          '},  // param_offsets',
          sep='', file=C_file)

    def make_mask(bit_number):
        r'Returns the mask as a C literal (in string form)'
        return hex(1 << (bit_number + 2))

    if param_block.optional:
        print(f'      {make_mask(param_block.kw_passed_bit)},  // keyword_mask',
              file=C_file)
    else:
        print('      0,  // keyword_mask', file=C_file)
    print('      (unsigned long []){',
          ', '.join(make_mask(mask) for mask in param_masks),
          '},  // param_masks',
          sep='', file=C_file)

    print('    },', file=C_file)


def get_uses(module):
    for v in module.names.values():
        if isinstance(v, symtable.Use):
            yield v


def get_labels(module):
    for v in module.names.values():
        if isinstance(v, symtable.Labeled_block):
            yield v


def gen_structs(module):
    emit_struct_start(module.C_struct_name)
    _emit_struct_variable("struct module_instance_s *", '__base__')
    for obj in module.names.values():
        if isinstance(obj, symtable.Use):
            _emit_struct_variable(f"struct {obj.module.C_struct_name} ",
                                  obj.C_local_name)
        elif isinstance(obj, symtable.Variable):
            emit_struct_variable(obj.type, obj.C_local_name, obj.dimensions)
        elif isinstance(obj, symtable.Labeled_block):
            _emit_struct_variable("unsigned long", obj.C_flags_name)
            if isinstance(obj, symtable.Subroutine):
                _emit_struct_variable("struct label_pointer_s",
                                      obj.C_ret_pointer_name)
            for type, name in obj.C_temps:
                emit_struct_variable(type, name)
            if obj.needs_dlt_mask:
                _emit_struct_variable('unsigned long', obj.C_dlt_mask_name)
    emit_struct_end()


def gen_instances(module):
    for path in gen_module_instances(module):
        yield f"{module.C_global_name}{path}"


def gen_module_instances(module):
    for use in get_uses(module):
        for path in gen_module_instances(use.module):
            yield f".{use.C_local_name}{path}"
    yield ''


def gen_init_fns(module):
    print(file=C_file)
    print("void", file=C_file)
    print(module.C_init_C_fn_name,
          '(struct ', module.C_struct_name, ' *module_instance',
          sep='', end='', file=C_file)
    if isinstance(module, symtable.Module):
        def gen_params(pb):
            for p in pb.required_params:
                print(', ', p.C_type, ' ', p.C_local_name,
                      sep='', end='', file=C_file)
            for p in pb.optional_params:
                print(', ', p.C_type, ' ', p.C_local_name,
                      sep='', end='', file=C_file)

        pb = module.pos_param_block
        if pb is not None:
            gen_params(pb)
        for pb in module.kw_parameters.values():
            gen_params(pb)
    print(") {", file=C_file)

    # FIX: Init module params/variables

    for use in get_uses(module):
        gen_module_init_call(use)

    for label in get_labels(module):
        gen_statement(
          f'module_instance->{sub.C_local_name}.__base__.descriptor = '
          f'&{sub.C_descriptor_name};')
        gen_statement(
          f'module_instance->{sub.C_local_name}.__base__.flags = 0;')
        gen_statement(
          f'module_instance->{sub.C_local_name}.__base__.global = '
          'module_instance;')

    print("}", file=C_file)


def init_descriptors(module):
    for label in get_labels(module):
        print(prepare_statement(
                f"{label.C_descriptor_name}.start_label = "
                f"&&{label.C_label_name};"),
              file=C_file)


def gen_module_init_call(use):
    print(file=C_file)
    print(prepare_statement(f"{use.module.C_init_C_fn_name}("
                            f"&module_instance->{use.C_local_name}"),
          end='', file=C_file)
    # FIX: Pass module args
    print(');', file=C_file)


def emit_param_blocks(obj):
    if obj.pos_param_block is not None:
        emit_param_block(obj.pos_param_block)
    for pb in obj.kw_parameters.values():
        emit_param_block(pb)


def emit_param_block(pb):
    for p in pb.required_params:
        emit_struct_variable(p.type, p.C_local_name, p.dimensions)
    if pb.optional_params:
        _emit_struct_variable(
          f"struct {pb.optional_param_block_struct_name }", 
          pb.optional_param_block_name)


def gen_instance(module):
    print(file=C_file)
    print(f"struct {module.C_struct_name} {module.C_global_name};", file=C_file)


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

    for label in get_labels(m):
        gen_label(label.C_label_name)

        if label.block is not None:
            lines.extend(gen_block(label.block))

    return lines


def gen_block(block):
    lines = []
    for s in block.statements:
        if isinstance(s, symtable.Statement):
            lines.append(prepare_statement(
                           f"current_module->lineno = {s.lineno};"))
            # FIX: gen code for s
        else:
            assert isinstance(s, symtable.DLT)
            # FIX: gen code for s
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
                if isinstance(var, symtable.Variable):
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


def emit_struct_variable(type, name, dimensions=()):
    _emit_struct_variable(translate_type(type), name, dimensions)


def _emit_struct_variable(type, name, dimensions=()):
    print(symtable.indent_str(Struct_var_indent), type,
          sep='', end='', file=C_file)
    if not type.endswith('*'):
        print(' ', end='', file=C_file)
    print(f"{name}", end='', file=C_file)
    for dim in dimensions:
        print(f"[{dim}]", end='', file=C_file)
    print(';', file=C_file)


def emit_struct_end():
    print(symtable.indent_str(Struct_indent), "};", sep='', file=C_file)


def emit_code_start():
    print(file=C_file)
    print(file=C_file)
    print("int", file=C_file)
    print("main(int argc, char **argv, char **env) {", file=C_file)
    print(symtable.indent_str(Statement_indent),
          "struct module_instance_s *current_module;",
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


def gen_statement(line):
    print(prepare_statement(line), file=C_file)


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
        assert name == m.name

    # {parent: set(uses)}
    order = dict(gen_dependencies(opmode))
    #print("tsort_modules, order:", order)
    for uses in order.values():
        for use in uses:
            assert use in order

    builtins = opmode.modules_seen['builtins']

    while order:
        leaves = set(parent
                     for parent, uses in order.items()
                      if not uses)
        if len(leaves) > 1 and builtins in leaves:
            leaves.remove(builtins)
        if len(leaves) > 1 and opmode in leaves:
            leaves.remove(opmode)
        for module in sorted(leaves, key=lambda m: m.name):
            yield module
            del order[module]
        for uses in order.values():
            uses -= leaves
    #yield opmode


def gen_dependencies(opmode):
    modules_seen = opmode.modules_seen
    all_modules = tuple(modules_seen.values()) + (opmode,)
    for m in all_modules:
        uses = set(modules_seen[x.module_name.value] for x in get_uses(m))
        yield m, uses
