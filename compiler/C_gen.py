# C_gen.py

import sys
import os.path
import operator
from functools import partial

import symtable
from scanner import Token, syntax_error
from code_generator import (
    todo, todo_with_args, translate_name, translate_type, unary, binary,
    subscript, relative_path, wrap,
)


NUM_FLAG_BITS = 64


Struct_indent = 0
Struct_var_indent = 4
Label_indent = 2
Statement_indent = 4


Assign_names = []
Gen_step_code = []
Module_instance_definitions = []
Module_instance_declarations = []
Label_descriptors = []
Module_descriptors = []
Module_instances = []    # list of lines (without final commas)
Init_labels = []
Set_module_descriptor_pointers = []
Initialize_module_parameters = []
Module_code = []


def start_output():
    "Opens C_file and writes the C_preamble to it."
    global C_file
    filename = f"{Opmode.name}.c"
    C_file = open(filename, 'wt')
    with open("C_preamble") as preamble:
        print(f"// {filename}", file=C_file)
        print(file=C_file)
        C_file.write(preamble.read())


def write_module_instance_list():
    print(file=C_file)
    print("const int __num_modules__ = ", len(Module_instances), ";",
          sep='', file=C_file)
    print("struct module_descriptor_s *__module_instances__[] = {", file=C_file)
    for line in Module_instances:
        print("    (struct module_descriptor_s *)&", line, ",",
              sep='', file=C_file)
    print("};", file=C_file)


def start_main():
    print(file=C_file)
    print("int", file=C_file)
    print("main(int argc, char *argv[], char *env[]) {", file=C_file)


def call_cycle_run():
    # FIX
    print(file=C_file)
    print("    // FIX: call cycle.run", file=C_file)


def end_main():
    print("}", file=C_file)


def done():
    C_file.close()


@symtable.Opmode.as_pre_hook("link_uses")
def assign_module_names(m):
    # set C_global_name: globally unique name
    if isinstance(m, symtable.Module):
        m.C_global_name = \
          f"{translate_name(m.name)}__module__{m.instance_number}"
        if m.parent_module is not None and \
           isinstance(m.parent_module, symtable.Module):
            m.dotted_name = f"{m.parent_module.dotted_name}.{m.name}"
        else:
            m.dotted_name = m.name
    else:
        m.C_global_name = translate_name(m.name) + "__opmode"
        m.dotted_name = m.name
    m.C_descriptor_name = m.C_global_name + "__desc"
    m.C_label_descriptor_name = m.C_global_name + "__label_desc"
    m.C_struct_name = m.C_global_name + "__instance_s"

    # list of fns that write elements into the module struct definition


@todo_with_args(symtable.Opmode, "prepare_module", Module_instance_definitions)
def write_module_instance_definition(module):
    print(file=C_file)
    print(f"struct {module.C_struct_name} {{", file=C_file)
    print("    struct module_descriptor_s *descriptor;", file=C_file)
    for var in module.filter(symtable.Variable):
        dims = ''.join(f"[{d}]" for d in var.dimensions)
        print(f"    {translate_type(var.type)} {var.C_local_name}{dims};",
              file=C_file)
    for label in module.filter(symtable.Label):
        if label.needs_dlt_mask:
            print(f"    unsigned long {label.C_dlt_mask_name};", file=C_file)
    print("};", file=C_file)


@todo_with_args(symtable.Opmode, "prepare_module", Module_instance_declarations)
def write_module_instance_declaration(module):
    print(file=C_file)
    print(f"struct {module.C_struct_name} {module.C_global_name};", file=C_file)


@symtable.Opmode.as_post_hook("link_uses")
def add_module_instance(m):
    Module_instances.append(m.C_global_name)


@todo_with_args(symtable.Opmode, "prepare_module", Module_descriptors)
def write_module_descriptor(m):
    print(file=C_file)
    print(f"struct module_descriptor_s {m.C_descriptor_name} = {{", file=C_file)
    print(f'    "{m.dotted_name}",   // name', file=C_file)
    print(f'    "{relative_path(m.filename)}",   // filename', file=C_file)
    if isinstance(m, symtable.Module):
        print(f'    &{m.C_label_descriptor_name},   // module_params',
              file=C_file)
    else:
        print('    NULL,   // module_params', file=C_file)
    print('    0,   // vars_set', file=C_file)
    print('    0,   // lineno', file=C_file)
    labels = tuple(label for label in m.filter(symtable.Label)
                         if not label.hidden)
    print(f'    {len(labels)},   // num_labels', file=C_file)
    print('    {   // labels', file=C_file)
    for label in labels:
        print(f"      &{label.C_label_descriptor_name},", file=C_file)
    print('    }', file=C_file)
    print("};", file=C_file)


@todo_with_args(symtable.Opmode, "prepare_module",
                Set_module_descriptor_pointers)
def set_module_descriptor_pointer(m):
    print(f"    {m.C_global_name}.descriptor = &{m.C_descriptor_name};",
          file=C_file)


@todo_with_args(symtable.Module, "prepare_module", Init_labels)
def init_module_label(module):
    print(f"    {module.C_label_descriptor_name}.module = "
            f"(struct module_descriptor_s *)&{module.C_global_name};",
          file=C_file)


@todo_with_args(symtable.Module, "prepare_module", Label_descriptors)
def write_module_label_descriptor(module):
    write_label_descriptor(module, module)


@todo_with_args(symtable.Module, "prepare_module", Module_code)
def write_module_code(module):
    for step in module.steps:
        step.write_code(module)


def write_label_code(label, module):
    print(f"  {label.C_global_name}:", file=C_file)
symtable.Label.write_code = write_label_code


def write_statement_code(label, module):
    # FIX: Implement
    print("    // FIX statement", file=C_file)
symtable.Statement.write_code = write_statement_code


def write_dlt_code(label, module):
    # FIX: Implement
    print(f"    // FIX DLT", file=C_file)
symtable.DLT.write_code = write_dlt_code


@symtable.Label.as_pre_hook("prepare")
def assign_label_names(label, module):
    # set C_global_name: globally unique name
    label.C_local_name = f"{translate_name(label.name)}__label"
    label.C_global_name = f"{module.C_global_name}__{label.C_local_name}"
    label.C_label_descriptor_name = label.C_global_name + "__desc"
    label.C_dlt_mask_name = f"{label.C_local_name}__dlt_mask"
    label.code = label.C_label_descriptor_name


@todo_with_args(symtable.Label, "prepare", Label_descriptors)
def write_label_label_descriptor(label, module):
    write_label_descriptor(label, module)


def write_label_descriptor(label, module):
    print(file=C_file)
    print(f"struct label_descriptor_s {label.C_label_descriptor_name} = {{",
          file=C_file)
    if isinstance(label, symtable.Module):
        print(f'    "{label.name}",   // name', file=C_file)
        print('    "module",   // type', file=C_file)
    else:
        print(f'    "{label.name.value}",   // name', file=C_file)
        print(f'    "{label.type.get_type().label_type}",   // type',
              file=C_file)
    param_blocks = tuple(label.gen_param_blocks())
    print(f'    {len(param_blocks)},   // num_param_blocks', file=C_file)
    print('    (struct param_block_descriptor_s []){   '
            '// param_block_descriptors',
          file=C_file)
    for pb in param_blocks:
        print("      {", file=C_file)
        if pb.name is None:
            print('        NULL,   // name', file=C_file)
        else:
            print(f'        "{pb.name}",   // name', file=C_file)
        print(f'        {len(pb.required_params) + len(pb.optional_params)},'
                '   // num_params',
              file=C_file)
        print(f'        {len(pb.required_params)},   // num_required_params',
              file=C_file)
        print(f'        {len(pb.optional_params)},   // num_optional_params',
              file=C_file)
        param_locations = ", ".join(
          f"&{module.C_global_name}.{p.variable.C_local_name}"
          for p in pb.gen_parameters())
        print(f'        (void *[]){{{param_locations}}},   // param_locations',
              file=C_file)
        var_set_masks = ", ".join(hex(1 << p.variable.set_bit)
                                  for p in pb.gen_parameters())
        print(f'        (unsigned long []){{{var_set_masks}}},'
                '// var_set_masks',
              file=C_file)
        if pb.optional:
            print(f'        {hex(1 << pb.kw_passed_bit)},   // kw_mask',
                  file=C_file)
        else:
            print('        0,   // kw_mask', file=C_file)
        param_masks = ", ".join(hex(1 << p.passed_bit)
                                for p in pb.optional_params)
        print(f'        (unsigned long []){{{param_masks}}},'
                '// param_masks',
              file=C_file)
        print("      },", file=C_file)
    print('    },', file=C_file)
    print('    0,   // params_passed', file=C_file)
    print("};", file=C_file)


@todo_with_args(symtable.Label, "prepare", Init_labels)
def init_label(label, module):
    print(f"    {label.C_label_descriptor_name}.module = "
            f"(struct module_descriptor_s *)&{module.C_global_name};",
          file=C_file)
    print(
      f"    {label.C_label_descriptor_name}.label = &&{label.C_global_name};",
      file=C_file)


@symtable.Variable.as_post_hook("prepare")
def assign_variable_names(var, module):
    var.C_local_name = translate_name(var.name)
    var.precedence, var.assoc = Precedence_lookup['.']
    var.code = f"{var.parent_namespace.C_global_name}.{var.C_local_name}"


@symtable.Literal.as_post_hook("prepare_step")
def compile_literal(literal, module, last_label, last_fn_subr):
    if isinstance(literal.value, (int, float)):
        literal.code = repr(literal.value)
    elif isinstance(literal.value, str):
        escaped_str = literal.value.replace('"', r'\"')
        literal.code = f'"{escaped_str}"'
    else:
        assert isinstance(literal.value, bool)
        literal.code = "1" if literal.value else "0"


@symtable.Subscript.as_post_hook("prepare_step")
def compile_subscript(subscript, module, last_label, last_fn_subr):
    subscript.precedence, subscript.assoc = Precedence_lookup['[']
    array_code = wrap(subscript.array_expr, subscript.precedence, 'left')
    sub_code = subscript.subscript_expr.get_step().code
    subscript.code = f"{array_code}[{sub_code}]"
    # FIX: add subscript range check to end of prep_statements


@symtable.Got_keyword.as_post_hook("prepare_step")
def compile_got_keyword(self, module, last_label, last_fn_subr):
    if self.immediate:
        self.precedence = 100
        self.assoc = None
        self.code = "1" if self.value else "0"
    else:
        self.precedence, self.assoc = Precedence_lookup['&']
        self.code = f"{self.label.C_label_descriptor_name}.params_passed" \
                    f" & {hex(1 << self.bit_number)}"


@symtable.Got_param.as_post_hook("prepare_step")
def compile_got_param(self, module, last_label, last_fn_subr):
    if self.immediate:
        self.precedence = 100
        self.assoc = None
        self.code = "1" if self.value else "0"
    else:
        self.precedence, self.assoc = Precedence_lookup['&']
        self.code = f"{self.label.C_label_descriptor_name}.params_passed" \
                    f" & {hex(1 << self.bit_number)}"


@symtable.Unary_expr.as_post_hook("prepare_step")
def compile_unary_expr(self, module, last_label, last_fn_subr):
    if self.immediate:
        self.precedence = 100
        self.assoc = None
        if isinstance(self.value, bool):
            self.code = "1" if self.value else "0"
        else:
            self.code = repr(self.value)
    else:
        self.precedence, self.assoc, self.code = \
          Unary_exprs[self.operator](self.expr)


@symtable.Binary_expr.as_post_hook("prepare_step")
def compile_binary_expr(self, module, last_label, last_fn_subr):
    if self.immediate:
        self.precedence = 100
        self.assoc = None
        if isinstance(self.value, bool):
            self.code = "1" if self.value else "0"
        elif isinstance(self.value, (int, float)):
            self.code = repr(self.value)
        else:
            escaped_str = self.value.replace('"', r'\"')
            self.code = f'"{escaped_str}"'
    else:
        self.precedence, self.assoc, self.code = \
          Binary_exprs[self.operator](self.expr1, self.expr2)


@symtable.Return_label.as_post_hook("prepare_step")
def compile_return_label(self, module, last_label, last_fn_subr):
    self.precedence, self.assoc = Precedence_lookup['.']
    self.code = f"{self.label.C_label_descriptor_name}.return_label"


Todo_lists = (
    Assign_names,
    Gen_step_code,
    [start_output],
    Module_instance_definitions,
    Module_instance_declarations,
    Label_descriptors,
    Module_descriptors,
    [write_module_instance_list, start_main],
    Init_labels,
    Set_module_descriptor_pointers,
    Initialize_module_parameters,
    [call_cycle_run],
    Module_code,
    [end_main, done],
)



Precedence = (  # highest to lowest
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


Reserved_words = frozenset((
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


def gen_abs(expr):
    precedence, assoc = Precedence_lookup['(']
    if expr.type.is_integer():
        return precedence, assoc, f"abs({expr})"
    return precedence, assoc, f"fabs({expr})"


Unary_exprs = {
    'ABS': gen_abs,
    'NOT': partial(unary, '!', '!{}'),
    '-': partial(unary, 'UMINUS', '-{}'),
}


Binary_exprs = {
    '.': partial(binary, '.', '{}.{}'),
    '^': partial(binary, ',', 'pow({}, {})', result_op='('),
    '*': partial(binary, '*', '{} * {}'),
    '/': partial(binary, '/', '{} / {}'),
    '//': partial(binary, '/', '{} / {}'),
    '%': partial(binary, '%', '{} % {}'),
    '+': partial(binary, '+', '{} + {}'),
    '-': partial(binary, '-', '{} - {}'),
    '<': partial(binary, '<', '{} < {}'),
    '<=': partial(binary, '<=', '{} <= {}'),
    '<~=': partial(binary, '-', '{} - {} < 1e-6', result_op='<'),
    '>': partial(binary, '>', '{} > {}'),
    '>=': partial(binary, '>=', '{} >= {}'),
    '>~=': partial(binary, '-', '{} - {} > -1e-6', result_op='>'),
    '==': partial(binary, '==', '{} == {}'),
    '~=': partial(binary, '-', 'fabs({} - {}) < 1e-6', result_op='<'),
    '!=': partial(binary, '!=', '{} != {}'),
    '<>': partial(binary, '!=', '{} != {}'),
    '!~=': partial(binary, '-', 'fabs({} - {}) >= 1e-6', result_op='>='),
    '<~>': partial(binary, '-', 'fabs({} - {}) >= 1e-6', result_op='>='),
    '[': subscript,
}


Types = {
    "float": "double",
    "integer": "long",
    "boolean": "char",
    "string": "char *",
    "module": "struct module_instance_s *",
}


def translate_label_type(type):
    return "struct label_descriptor_s *"


#####################################################################


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
        C_str = expr.replace('""', r'\"')
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

