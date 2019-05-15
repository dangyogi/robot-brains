# C_gen.py

import sys
import os.path
import operator
from functools import partial
from itertools import chain, groupby
from pkg_resources import resource_stream

from robot_brains import symtable
from robot_brains.scanner import Token, syntax_error
from robot_brains.code_generator import (
    todo, todo_with_args, translate_name, translate_type, unary, binary,
    subscript, relative_path, wrap, compile_unary, compile_binary,
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
    with resource_stream(__name__, "C_preamble") as preamble:
        print(f"// {filename}", file=C_file)
        print(file=C_file)
        C_file.write(preamble.read().decode('ascii'))


def write_module_instance_list():
    print(file=C_file)
    print("const int __num_modules__ = ", len(Module_instances), ";",
          sep='', file=C_file)
    print("struct module_descriptor_s *__module_instances__[] = {", file=C_file)
    for line in Module_instances:
        print("    &", line, ",", sep='', file=C_file)
    print("};", file=C_file)


def start_main():
    print(file=C_file)
    print("int", file=C_file)
    print("main(int argc, char *argv[], char *env[]) {", file=C_file)
    print("    my_set_signal(SIGINT, my_handler);", file=C_file)
    print("    my_set_signal(SIGTERM, my_handler);", file=C_file)
    # FIX: ignore SIGHUP?


@todo(Gen_step_code)
def create_cycle_irun_code():
    global Cycle_irun_code

    cycle = symtable.lookup(Token.dummy('cycle'), Opmode, error_not_found=False)
    if cycle is None or not isinstance(cycle.get_step(), symtable.Module):
        print("Could not find 'cycle' module", file=sys.stderr)
        sys.exit(1)
    cycle = cycle.get_step()
    run = cycle.lookup(Token.dummy('irun'), error_not_found=False)
    if run is None or not isinstance(run.get_step(), symtable.Function):
        print("Could not find 'irun' function in 'cycle' module",
              file=sys.stderr)
        sys.exit(1)
    run = run.get_step()
    if run.pos_param_block is not None or run.kw_parameters:
        print("'cycle.irun' must not take any parameters", file=sys.stderr)
        sys.exit(1)
    integer = symtable.Builtin_type('integer')
    if run.return_types != (((integer,), ()), ()):
        print("'cycle.irun' must only return an INTEGER", file=sys.stderr)
        sys.exit(1)
    var = run.create_variable('__exit_status__', integer)
    done_label = run.new_fn_ret_label(run.parent_namespace, var)
    terminate = cycle.lookup(Token.dummy('terminate'), error_not_found=False)
    if terminate is None or type(terminate.get_step()) is not symtable.Label:
        print("Could not find 'terminate' label in 'cycle' module",
              file=sys.stderr)
        sys.exit(1)
    terminate = terminate.get_step()
    if terminate.pos_param_block is None or \
       len(terminate.pos_param_block.required_params) != 1 or \
       terminate.pos_param_block.required_params[0].type != integer or \
       terminate.pos_param_block.optional_params or \
       terminate.kw_parameters:
        print("'cycle.terminate' must only take an INTEGER parameter",
              file=sys.stderr)
        sys.exit(1)
    Cycle_irun_code = (
        "    if (setjmp(longjmp_to_main)) {",
        "        longjmp_to_main_set = 1;",
        f"        {terminate.C_label_descriptor_name}.params_passed = 0;",
        f"        *(int *)param_location({terminate.code}, NULL, 0) = " \
                      "Exit_status;",
        f"        goto *{terminate.C_label_descriptor_name}.label;",
        "    }",
        f"    {run.C_label_descriptor_name}.params_passed = FLAG_RUNNING;",
        f"    {run.C_label_descriptor_name}.return_label = " \
                f"&{done_label.C_label_descriptor_name};",
        f"    goto *{run.C_label_descriptor_name}.label;",
        f"",
        f"  {done_label.decl_code}",
        f"    return {var.code};",
    )


def call_cycle_irun():
    print(file=C_file)
    print("    // call cycle.irun", file=C_file)
    for line in Cycle_irun_code:
        print(line, file=C_file)
    print(file=C_file)


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
    m.code = '&' + m.C_descriptor_name

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
    Module_instances.append(m.C_descriptor_name)


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
            f"&{module.C_descriptor_name};",
          file=C_file)


@todo_with_args(symtable.Module, "prepare_module", Label_descriptors)
def write_module_label_descriptor(module):
    write_label_descriptor(module, module)


@todo_with_args(symtable.Module, "prepare_module", Module_code)
def write_module_code(module):
    for step in module.steps:
        step.write_code(module)


@todo_with_args(symtable.Use, "prepare_used_module",
                Initialize_module_parameters)
def write_use_args(self, module):
    write_args('&' + self.module.C_label_descriptor_name,
               self.module.params_type, self.pos_arguments, self.kw_arguments)


def write_label_code(label, module, extra_indent=0):
    print(' ' * extra_indent, f"  {label.decl_code}", sep='', file=C_file)
symtable.Label.write_code = write_label_code


def write_statement_code(self, module, extra_indent=0):
    indent = extra_indent + 4
    print(' ' * indent,
          f"{self.containing_label.C_label_descriptor_name}.lineno"
            f" = {self.lineno};",
          sep='', file=C_file)
    check_all_vars_used(self.containing_label, self.vars_used, extra_indent)
    write_prep_statements(self.prep_statements, extra_indent)
    self.write_check_running(extra_indent)

    self.write_code_details(module, extra_indent)

    for line in self.post_statements:
        print(' ' * indent, line, sep='', file=C_file)
symtable.Statement.write_code = write_statement_code


def write_prep_statements(prep_statements, extra_indent):
    indent = extra_indent + 4
    for line in prep_statements:
        if isinstance(line, str):
            print(' ' * indent, line, sep='', file=C_file)
        else:
            line(extra_indent)


def check_statement_running(self, extra_indent):
    pass
symtable.Statement.write_check_running = check_statement_running


def check_all_vars_used(containing_label, vars_used, extra_indent):
    for vars_module, module_vars_used \
     in groupby(sorted(vars_used,
                       key=lambda v: (id(v.parent_namespace), v.set_bit)),
                key=lambda v: v.parent_namespace):
        check_vars_used(containing_label, vars_module, module_vars_used,
                        extra_indent)


def check_vars_used(label, vars_module, vars_used, extra_indent):
    indent = extra_indent + 4
    vars_used = tuple(vars_used)
    mask = sum(1 << v.set_bit for v in vars_used)
    bits_set = f"{vars_module.C_descriptor_name}.vars_set & {hex(mask)}"
    print(' ' * indent, f"if ({bits_set} != {hex(mask)}) {{",
          sep='', file=C_file)
    print(' ' * indent,
          f"    report_var_not_set(&{label.C_label_descriptor_name},",
          sep='', file=C_file)
    print(' ' * indent,
          f"                       {bits_set}, {hex(mask)},",
          sep='', file=C_file)
    var_names = ', '.join(f'"{v.name.value}"' for v in vars_used)
    print(' ' * indent, f"                       {var_names});",
          sep='', file=C_file)
    print(' ' * indent, "}", sep='', file=C_file)


def write_continue_details(self, module, extra_indent):
    print(' ' * extra_indent, "    break;", sep='', file=C_file)
symtable.Continue.write_code_details = write_continue_details


def write_set_details(self, module, extra_indent):
    indent = extra_indent + 4
    print(' ' * indent,
          f"{self.lvalue.code} = {self.primary.code};",
          sep='', file=C_file)
    var_set = self.lvalue.var_set
    if var_set is not None:
        module_desc = f"{var_set.parent_namespace.C_descriptor_name}"
        print(' ' * indent,
              f"{module_desc}.vars_set |= {hex(1 << var_set.set_bit)};",
              sep='', file=C_file)
symtable.Set.write_code_details = write_set_details


def write_native_details(self, module, extra_indent):
    indent = extra_indent + 4
    print(' ' * indent, self.code, sep='', file=C_file)
symtable.Native_statement.write_code_details = write_native_details


def write_args(dest_label, label_type, pos_args, kw_args, extra_indent=0):
    indent = extra_indent + 4
    def write_pb(req_types, opt_types, args, pb_name=None):
        if pb_name is None:
            keyword = 'NULL'
        else:
            keyword = f'"{pb_name}"'
        for i, (type, arg) in enumerate(zip(chain(req_types, opt_types), args)):
            print(' ' * indent,
                  f'*({translate_type(type)} *)'
                  f'param_location({dest_label}, {keyword}, {i}) = '
                  f'{arg.get_step().code};',
                  sep='', file=C_file)
    write_pb(label_type.required_params, label_type.optional_params, pos_args)
    for keyword, args in kw_args:
        opt, req_types, opt_types = label_type.kw_params[keyword]
        write_pb(req_types, opt_types, args, keyword.value)


def write_goto_details(self, module, extra_indent):
    indent = extra_indent + 4
    primary = self.primary.get_step()
    label = primary.code
    print(' ' * indent, f'({label})->params_passed = 0;', sep='', file=C_file)
    write_args(label, primary.type.get_type(),
               self.arguments[0], self.arguments[1], extra_indent)
    print(' ' * indent, f'goto *({label})->label;', sep='', file=C_file)
symtable.Goto.write_code_details = write_goto_details


def write_return_details(self, module, extra_indent):
    indent = extra_indent + 4
    write_done(self.from_opt, self.containing_label, extra_indent)
    print(' ' * indent,
          f"({self.dest_label})->params_passed = 0;",
          sep='', file=C_file)
    write_args(self.dest_label, self.label_type,
               self.arguments[0], self.arguments[1], extra_indent)
    print(' ' * indent, f'goto *({self.dest_label})->label;',
          sep='', file=C_file)
symtable.Return.write_code_details = write_return_details


def check_call_running(self, extra_indent):
    indent = extra_indent + 4
    label = self.primary.get_step().code
    print(' ' * indent,
          f"if (({label})->params_passed & FLAG_RUNNING) {{",
          sep='', file=C_file)
    print(' ' * (indent + 4),
          f"report_error(&{self.containing_label.C_label_descriptor_name},",
          sep='', file=C_file)
    print(' ' * (indent + 4),
          f'             "\'%s\' still running\\n", ({label})->name);',
          sep='', file=C_file)
    print(' ' * indent, "}", sep='', file=C_file)
    print(' ' * indent,
          f"({label})->params_passed = FLAG_RUNNING;",
          sep='', file=C_file)
symtable.Call_statement.write_check_running = check_call_running
symtable.Call_fn.write_check_running = check_call_running

def write_call_details(self, module, extra_indent):
    indent = extra_indent + 4
    primary = self.primary.get_step()
    label = primary.code
    write_args(label, primary.type.get_type(),
               self.arguments[0], self.arguments[1], extra_indent)
    print(' ' * indent,
          f'({label})->return_label = {self.returning_to.code};',
          sep='', file=C_file)
    print(' ' * indent, f'goto *({label})->label;',
          sep='', file=C_file)
symtable.Call_statement.write_code_details = write_call_details


def write_opeq_details(self, module, extra_indent):
    indent = extra_indent + 4
    _, _, code = compile_binary(self.lvalue, self.operator.split('=')[0],
                                self.expr)
    print(' ' * indent, f"{self.lvalue.code} = {code};", sep='', file=C_file)
symtable.Opeq_statement.write_code_details = write_opeq_details


def write_done_details(self, module, extra_indent):
    write_done(self.label, self.containing_label, extra_indent)
symtable.Done_statement.write_code_details = write_done_details


def write_done(done_label, containing_label, extra_indent):
    indent = extra_indent + 4
    done_label_code = done_label.get_step().code
    extra_indent += 4
    print(' ' * indent,
          f"if (!(({done_label_code})->params_passed & FLAG_RUNNING)) {{",
          sep='', file=C_file)
    print(' ' * (indent + 4),
          f"report_error(&{containing_label.C_label_descriptor_name},",
          sep='', file=C_file)
    print(' ' * (indent + 4),
          '            "\'%s\' not running -- second return?\\n",',
          sep='', file=C_file)
    print(' ' * (indent + 4),
          f'             ({done_label_code})->name);',
          sep='', file=C_file)
    print(' ' * indent, "}", sep='', file=C_file)
    print(' ' * indent,
          f"({done_label_code})->params_passed &= ~FLAG_RUNNING;",
          sep='', file=C_file)


def write_dlt_code(self, module, extra_indent=0):
    indent = extra_indent + 4
    label = self.conditions.containing_label
    print(' ' * indent, f"{label.C_label_descriptor_name}.lineno"
            f" = {self.conditions.lineno};",
          sep='', file=C_file)
    check_all_vars_used(label, self.conditions.vars_used, extra_indent)
    write_prep_statements(self.conditions.prep_statements, extra_indent)
    dlt_mask = f"{module.C_global_name}.{label.C_dlt_mask_name}"
    print(' ' * indent, f"{dlt_mask} = 0;", sep='', file=C_file)
    for i, expr in enumerate(reversed(self.conditions.exprs)):
        print(' ' * indent, f"if ({expr.code}) {{",
              sep='', file=C_file)
        print(' ' * indent, f"    {dlt_mask} |= {hex(1 << i)};",
              sep='', file=C_file)
        print(' ' * indent, "}", sep='', file=C_file)
    print(' ' * indent, f"switch ({dlt_mask}) {{",
          sep='', file=C_file)
    for action in self.actions.actions:
        action.write_code(module, extra_indent + 4)
    print(' ' * indent, "}", sep='', file=C_file)
symtable.DLT.write_code = write_dlt_code


def write_dlt_map_code(self, module, extra_indent=0):
    for mask in self.masks:
        print(' ' * extra_indent, f"case {hex(mask)}:", sep='', file=C_file)
symtable.DLT_MAP.write_code = write_dlt_map_code


@symtable.Label.as_pre_hook("prepare")
def assign_label_names(label, module):
    # set C_global_name: globally unique name
    label.C_local_name = f"{translate_name(label.name)}__label"
    label.C_global_name = f"{module.C_global_name}__{label.C_local_name}"
    label.C_label_descriptor_name = label.C_global_name + "__desc"
    label.C_dlt_mask_name = f"{label.C_local_name}__dlt_mask"
    label.code = '&' + label.C_label_descriptor_name
    label.decl_code = label.C_global_name + ':'


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
        elif pb.optional:

            print(f'        "{pb.name.value[1:]}",   // name', file=C_file)
        else:
            print(f'        "{pb.name.value}",   // name', file=C_file)
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
    print('    0,   // lineno', file=C_file)
    print("};", file=C_file)


@todo_with_args(symtable.Label, "prepare", Init_labels)
def init_label(label, module):
    print(f"    {label.C_label_descriptor_name}.module = "
            f"&{module.C_descriptor_name};",
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
    if isinstance(literal.value, bool):
        literal.code = "1" if literal.value else "0"
    elif isinstance(literal.value, (int, float)):
        literal.code = repr(literal.value)
    else:
        assert isinstance(literal.value, str)
        escaped_str = literal.value.replace('"', r'\"')
        literal.code = f'"{escaped_str}"'


@symtable.Subscript.as_post_hook("prepare_step")
def compile_subscript(subscript, module, last_label, last_fn_subr):
    subscript.precedence, subscript.assoc = Precedence_lookup['[']
    array_code = wrap(subscript.array_expr, subscript.precedence, 'left')
    sub_codes = [sub_expr.get_step().code
                 for sub_expr in subscript.subscript_exprs]
    subscripts = ''.join(f"[{sub}]" for sub in sub_codes)
    subscript.code = f"{array_code}{subscripts}"
    subscript.prep_statements += tuple(
      f'range_check(&{subscript.containing_label.C_label_descriptor_name}, '
                  f'{sub}, {dim});'
      for dim, sub in zip(subscript.array_expr.get_step().dimensions, sub_codes)
    )


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


@symtable.Call_fn.as_post_hook("prepare_step")
def compile_call_fn(self, module, last_label, last_fn_subr):
    self.prep_statements = self.prep_statements + (self.write_call,)


def write_call_fn_call(self, extra_indent):
    indent = extra_indent + 4
    self.write_check_running(extra_indent)
    primary = self.primary.get_step()
    label = primary.code
    write_args(label, primary.type.get_type(),
               self.pos_arguments, self.kw_arguments, extra_indent)
    print(' ' * indent, f'({label})->return_label = {self.ret_label.code};',
          sep='', file=C_file)
    print(' ' * indent, f'goto *({label})->label;',
          sep='', file=C_file)
    print(' ' * (indent - 2), f'{self.ret_label.decl_code}',
          sep='', file=C_file)
symtable.Call_fn.write_call = write_call_fn_call


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
          compile_unary(self.operator, self.expr)


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
            assert isinstance(self.value, str)
            escaped_str = self.value.replace('"', r'\"')
            self.code = f'"{escaped_str}"'
    else:
        self.precedence, self.assoc, self.code = \
          compile_binary(self.expr1, self.operator, self.expr2)


@symtable.Return_label.as_post_hook("prepare_step")
def compile_return_label(self, module, last_label, last_fn_subr):
    self.precedence, self.assoc = Precedence_lookup['.']
    self.code = f"{self.label.C_label_descriptor_name}.return_label"


Todo_lists = (
    ("Assign_names", Assign_names),
    ("Gen_step_code", Gen_step_code),
    ("start_output", [start_output]),
    ("Module_instance_definitions", Module_instance_definitions),
    ("Module_instance_declarations", Module_instance_declarations),
    ("Label_descriptors", Label_descriptors),
    ("Module_descriptors", Module_descriptors),
    ("write_module_instance_list/start_main",
     [write_module_instance_list, start_main]),
    ("Init_labels", Init_labels),
    ("Set_module_descriptor_pointers", Set_module_descriptor_pointers),
    ("Initialize_module_parameters", Initialize_module_parameters),
    ("call_cycle_irun", [call_cycle_irun]),
    ("Module_code", Module_code),
    ("end_main/done", [end_main, done]),
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
        return precedence, assoc, f"abs({expr.get_step().code})"
    return precedence, assoc, f"fabs({expr.get_step().code})"


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
    "module": "struct module_descriptor_s *",
}


def translate_label_type(type):
    return "struct label_descriptor_s *"

