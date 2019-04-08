# symtable.py

from collections import OrderedDict

import scanner


def push_namespace(ns):
    # No longer used...
    scanner.Namespaces.append(ns)
    return entity


def pop_namespace():
    del scanner.Namespaces[-1]


def current_namespace():
    return scanner.Namespaces[-1]


def top_namespace():
    return scanner.Namespaces[0]


class Symtable:
    r'''Root of all symtable classes.
    '''
    def dump(self, f, indent=0):
        print(f"{indent_str(indent)}{self.__class__.__name__} ", end='', file=f)
        self.dump_details(f)
        print(file=f)
        self.dump_contents(f, indent + 2)

    def dump_details(self, f):
        pass

    def dump_contents(self, f, indent):
        pass


class Entity(Symtable):
    r'''Named entities within the current namespace.
    '''
    def __init__(self, ident, no_prior_use=False):
        self.name = ident
        self.set_in_namespace(no_prior_use)

    def set_in_namespace(self, no_prior_use):
        current_namespace().set(self.name, self, no_prior_use)

    def dump_details(self, f):
        super().dump_details(f)
        print(f" {self.name.value}", end='', file=f)

    def prepare(self):
        pass

    def gen_prep(self, generator):
        #print(f"{self.name.value}.gen_prep")
        self.gen_name = self.name.value

    def gen_structs(self, generator):
        #print(f"{self.name.value}.gen_structs -- ignored")
        pass

    def gen_struct_entries(self, generator):
        #print(f"{self.name.value}.gen_struct_entries -- ignored")
        pass

    def gen_struct_entry(self, generator):
        #print(f"{self.name.value}.gen_struct_entry -- ignored")
        pass

    def gen_code(self, generator):
        print(f"{self.name.value}.gen_structs -- ignored")
        pass


class Reference(Entity):
    def __init__(self, ident, **kws):
        Entity.__init__(self, ident)
        for k, v in kws.items():
            setattr(self, k, v)


class Variable(Entity):
    assignment_seen = False
    dim = None

    def __init__(self, ident, *, no_prior_use=False, **kws):
        Entity.__init__(self, ident, no_prior_use)
        self.type = ident.type
        for k, v in kws.items():
            setattr(self, k, v)

    def dump_details(self, f):
        super().dump_details(f)
        print(f" type={self.type}", end='', file=f)

    def gen_struct_entry(self, generator):
        #print(self.name.value, "gen_struct_entry", self.assignment_seen)
        if self.assignment_seen:
            generator.emit_struct_variable(self.type, self.name.value, self.dim)


class Required_parameter(Variable):
    assignment_seen = True

    def __init__(self, ident, **kws):
        Variable.__init__(self, ident, **kws)
        current_namespace().pos_parameter(self)

    def dump_details(self, f):
        super().dump_details(f)
        if hasattr(self, 'param_pos_number'):
            print(f" param_pos_number={self.param_pos_number}", end='', file=f)


class Optional_parameter(Required_parameter):
    def __init__(self, ident, default, **kws):
        self.default = default
        Required_parameter.__init__(self, ident, **kws)

    def dump_details(self, f):
        super().dump_details(f)
        print(f" default={self.default}", end='', file=f)


class Use(Entity):
    def __init__(self, ident, module_name, arguments):
        Entity.__init__(self, ident)
        self.module_name = module_name
        self.arguments = arguments

    def dump_details(self, f):
        super().dump_details(f)
        print(f" module_name={self.module_name}", end='', file=f)


class Typedef(Entity):
    r'''definition is either:

       (FUNCTION, taking_opt, returning_opt)
    or (SUBROUTINE, taking_opt)
    '''
    def __init__(self, ident, definition):
        Entity.__init__(self, ident)
        self.definition = definition

    def dump_details(self, f):
        super().dump_details(f)
        print(f" definition={self.definition}", end='', file=f)


class With_parameters:
    def __init__(self):
        self.current_parameters = self.pos_parameters = []
        self.kw_parameters = OrderedDict()
        self.seen_default = False

    def pos_parameter(self, param):
        if isinstance(param, Required_parameter):
            if self.seen_default:
                scanner.syntax_error("Required parameter may not follow "
                                     "default parameter",
                                     ident.lexpos,
                                     ident.lineno)
        else:
            self.seen_default = True
        param.param_pos_number = len(self.current_parameters)
        self.current_parameters.append(param)

    def kw_parameter(self, keyword):
        lname = keyword.value.lower()
        if lname in self.kw_parameters:
            scanner.syntax_error("Duplicate keyword",
                                 keyword.lexpos, keyword.lineno)
        self.current_parameters = self.kw_parameters[lname] = []
        self.seen_default = False

    def prepare(self):
        self.num_pos_defaults = sum(1 for p in self.pos_parameters
                                      if isinstance(p, Optional_parameter))
        self.num_kw_defaults = {kw: sum(1 for p in params
                                          if isinstance(p, Optional_parameter))
                                for kw, params in self.kw_parameters.items()}
        self.no_kw_defaults = not any(num
                                      for num in self.num_kw_defaults.values())

    def dump_details(self, f):
        super().dump_details(f)
        print(f" pos_params {self.pos_parameters}"
                f" kw_params {self.kw_parameters.items()}",
              end='',
              file=f)


class Labeled_block(Entity, With_parameters):
    def __init__(self, ident):
        self.parent_namespace = current_namespace()
        Entity.__init__(self, ident)
        With_parameters.__init__(self)
        current_namespace().push_labeled_block(self)
        self.code_name = f"{self.parent_namespace.code_name}__{self.name.value}"

    def add_first_block(self, block):
        self.block = block

    def dump_details(self, f):
        super().dump_details(f)
        print(f" block {self.block}", end='', file=f)

    def gen_code(self, generator):
        generator.emit_label(self.code_name)
        self.block.gen_code(generator)


class Namespace(Entity):
    def __init__(self, ident):
        Entity.__init__(self, ident)
        self.names = OrderedDict()
        #print(f"scanner.Namespaces.append({ident})")
        scanner.Namespaces.append(self)
        self.code_name = self.name.value

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.name.value}>"

    def lookup(self, ident, error_not_found=True):
        lname = ident.value.lower()
        if lname not in self.names:
            if error_not_found:
                syntax_error("Undefined name", ident.lexpos, ident.lineno)
            return None
        return self.names[lname]

    def set(self, ident, entity, no_prior_use=False):
        lname = ident.value.lower()
        current_entity = self.names.get(lname)
        if current_entity is not None and \
           (no_prior_use or not isinstance(current_entity, Reference)):
            scanner.syntax_error("Duplicate definition",
                                 ident.lexpos, ident.lineno)
        self.names[lname] = entity

    def ident_ref(self, ident):
        lname = ident.value.lower()
        if lname not in self.names:
            self.names[lname] = Reference(ident)
        return self.names[lname]

    def assigned_to(self, ident):
        lname = ident.value.lower()
        current_entity = self.names.get(lname)
        if current_entity is None or isinstance(current_entity, Reference):
            self.names[lname] = Variable(ident, assignment_seen=True)
            return
        if isinstance(current_entity, Variable):
            current_entity.assignment_seen = True
        else:
            syntax_error(f"Can not set {current_entity.__class__.__name__}",
                         ident.lexpos, indent.lineno)

    def dump_contents(self, f, indent):
        for entity in self.names.values():
            entity.dump(f, indent)

    def gen_prep(self, generator):
        for entity in self.names.values():
            entity.gen_prep(generator)

    def gen_structs(self, generator):
        #print("Namespace", self.name.value, "gen_structs")
        for entity in self.names.values():
            entity.gen_structs(generator)

    def gen_struct_entries(self, generator):
        #print("Namespace", self.name.value, "gen_struct_entries")
        for entity in self.names.values():
            entity.gen_struct_entry(generator)

    def gen_code(self, generator):
        print("Namespace", self.name.value, "gen_code")
        for entity in self.names.values():
            entity.gen_code(generator)


class Dummy_token:
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"<Dummy_token {self.value!r}>"


class Opmode(Namespace):
    def __init__(self, modules_seen=None):
        Namespace.__init__(self, Dummy_token(scanner.file_basename()))
        self.filename = scanner.filename()
        if modules_seen is not None:
            self.modules_seen = modules_seen

    def set_in_namespace(self, no_prior_use):
        pass

    def gen_program(self, generator):
        self.gen_prep(generator)
        self.gen_structs(generator)
        generator.start_code()
        self.gen_code(generator)
        generator.end_code()


class Module(Opmode, With_parameters):
    def __init__(self):
        Opmode.__init__(self)
        With_parameters.__init__(self)
        self.struct_name = self.name.value + "_module"

    def gen_structs(self, generator):
        super().gen_structs(generator)
        self.type_name = generator.emit_struct_start(self.struct_name)
        self.gen_struct_entries(generator)
        generator.emit_struct_end()


class Subroutine(Namespace, With_parameters):
    struct_name_suffix = "_sub"
    dlt_mask_needed = False
    first_block = None

    def __init__(self, ident):
        self.parent_namespace = current_namespace()
        Namespace.__init__(self, ident)
        With_parameters.__init__(self)
        self.current_block = None
        self.code_name = f"{self.parent_namespace.code_name}__{self.name.value}"
        self.struct_name = self.code_name + self.struct_name_suffix
        self.vars_name = self.name.value + "_vars"
        self.labels = {}        # {label: parameters} only with parameters
        self.start_labels = {}  # {base_name: [label]}

    def add_first_block(self, first_block):
        if self.current_block is None:
            self.first_block = first_block
        else:
            self.current_block.add_first_block(first_block)

    def push_labeled_block(self, labeled_block):
        self.current_block = labeled_block

    def dump_contents(self, f, indent):
        super().dump_contents(f, indent)
        self.first_block.dump(f, indent)

    #def gen_prep(self, generator):
    #    self.gen_name = self.name.value

    def gen_structs(self, generator):
        self.type_name = generator.emit_struct_start(self.struct_name)
        self.gen_ret_entry(generator)
        self.gen_struct_entries(generator)
        generator.emit_struct_end()

    def gen_ret_entry(self, generator):
        generator.emit_struct_sub_ret_info()

    def gen_struct_entry(self, generator):
        generator._emit_struct_variable(self.type_name, self.vars_name)

    def gen_code(self, generator):
        pass


class Function(Subroutine):
    struct_name_suffix = "_fn"

    def __init__(self, ident):
        Subroutine.__init__(self, ident)
        self.return_types = ident.type,

    def set_return_types(self, return_types):
        self.return_types = return_types

    def dump_details(self, f):
        super().dump_details(f)
        print(f" return_types {self.return_types}", end='', file=f)

    def gen_ret_entry(self, generator):
        generator.emit_struct_fn_ret_info()


class Block(Symtable):
    def __init__(self, statements):
        self.statements = statements

    def dump_contents(self, f, indent):
        for s in self.statements:
            s.dump(f, indent)

    def gen_code(self, f, indent=0):
        for s in self.statements:
            s.gen_code(f, indent)


class Statement(Symtable):
    arguments = {
        'pass': (),
        'prepare': ('expr', (('*', 'expr'), ('*', ('token', ('*', 'expr'))))),
        'reuse': ('expr', 'ignore', 'expr'),
        'release': ('expr'),
        'set': ('lvalue', 'ignore', 'expr'),
        'goto': ('expr', ('*', 'expr'), 'ignore', 'expr'),
        'return': (('*', 'expr'), 'ignore', 'expr'),
    }

    formats = {
        'pass': (),
        'prepare': ('prepare {}',),
        'reuse': ('reuse {}',),
        'release': ('release {}',),
        'set': ('{0[0]} = {0[2]}',),
        'goto': ('goto {}',),
        'return': ('return {}',),
    }

    def __init__(self, *args):
        self.args = args

    def dump_details(self, f):
        super().dump_details(f)
        print(f" {self.args}", end='', file=f)

    def gen_code(self, generator, indent):
        generator.reset_temps()

        statements = []
        def process_expr(arg):
            pre_statements, C_expr, _, _ = generator.gen_expr(arg)
            statements.extend(pre_statements)
            return C_expr

        def process_lvalue(arg):
            pre_statements, C_lvalue, _, _ = generator.gen_lvalue(arg)
            statements.extend(pre_statements)
            return C_lvalue

        statement_type = self.args[0].lower()

        processed_args = self.follow(self.args,
                                     self.arguments[statement_type],
                                     process_expr,
                                     process_lvalue)

        statements.extend(format.format(processed_args)
                          for format in self.formats[statement_type])
        generator.emit_statements(statements, indent)

    def follow(self, args, arg_map, expr_fn, lvalue_fn):
        if arg_map == 'expr':
            return expr_fn(args)
        if arg_map == 'lvalue':
            return lvalue_fn(args)
        if arg_map == 'ignore':
            return args
        assert isinstance(arg_map, tuple)
        assert isinstance(args, tuple)
        if arg_map[0] == '*':
            assert len(arg_map) == 2, \
                   f"expected len(arg_map) == 2 for {arg_map}"
            return tuple(self.follow(arg, arg_map[1], expr_fn, lvalue_fn)
                         for arg in args)
        assert len(args) <= len(arg_map)
        return tuple(self.follow(arg, map, expr_fn, lvalue_fn)
                     for arg, map in zip(args, arg_map))


class Call_statement(Statement):
    # primary arguments [RETURNING_TO expr]
    pass


class Opeq_statement(Statement):
    # primary OPEQ expr
    pass


class Conditions(Symtable):
    r'''DLT conditions section.

    self.exprs is condition expressions in order from MSB to LSB in the mask.
    self.num_columns is the number of columns in the DLT.
    self.offset is the number of spaces between the '|' and the first column.
    self.column_masks has one list of masks for each column.
    self.last_rest is the last rest Token (for future syntax errors).
    '''
    def __init__(self, conditions):
        r'self.conditions is ((expr, rest), ...)'
        self.compile_conditions(conditions)

    def compile_conditions(self, conditions):
        self.exprs = []  # exprs form bits in mask, ordered MSB to LSB
        self.offset = None
        for expr, rest in conditions:
            if not rest.value.strip():
                scanner.syntax_error("Missing condition flags",
                                     rest.lexpos, rest.lineno)
            if self.offset is None:
                self.offset = len(rest.value) - len(rest.value.lstrip())
                self.num_columns = len(rest.value.strip())
                columns = [[] for _ in range(self.num_columns)]
            this_offset = len(rest.value) - len(rest.value.lstrip())
            if this_offset != self.offset:
                scanner.syntax_error("Condition flags don't start in the same "
                                     "column as previous condition",
                                     rest.lexpos + this_offset, rest.lineno)
            rest.value = rest.value.lstrip()
            rest.lexpos += this_offset
            if len(rest.value.strip()) != self.num_columns:
                scanner.syntax_error("Must have same number of condition flags "
                                     "as previous condition",
                                     rest.lexpos, rest.lineno)
            self.exprs.append(expr)
            for i, flag in enumerate(rest.value.strip()):
                if flag not in '-yYnN':
                    scanner.syntax_error("Illegal condition flag, "
                                         "expected 'y', 'n', or '-'",
                                         rest.lexpos + i, rest.lineno)
                columns[i].append(flag.upper())

        #print("exprs", self.exprs, "offset", self.offset,
        #      "num_columns", self.num_columns, "columns", columns)

        self.last_rest = rest

        # Generate column_masks.  Column_masks has one list of masks for each
        # column.
        self.column_masks = []
        for flags in columns:
            self.column_masks.append(self.gen_masks(flags))
        #print("column_masks", self.column_masks)

        # Check for duplicate condition flags
        seen = set()
        for i, masks in enumerate(self.column_masks):
            for mask in masks:
                if mask in seen:
                    scanner.syntax_error("Duplicate condition flags",
                                         self.last_rest.lexpos + i,
                                         self.last_rest.lineno)
                else:
                    seen.add(mask)

        # Check to insure all condition combinations were specified
        all_masks = frozenset(range(2**len(self.exprs)))
        #print("all_masks", all_masks, "seen", seen)
        unseen = [self.decode_mask(missing)
                  for missing in sorted(all_masks - seen)]
        if unseen:
            scanner.syntax_error("Missing condition flags (top to bottom): "
                                   + ', '.join(unseen),
                                 self.last_rest.lexpos + self.num_columns,
                                 self.last_rest.lineno)

    def gen_masks(self, flags, i=None):
        if not flags: return [0]
        if i is None: i = len(flags) - 1
        masks = []
        if flags[0] in 'Y-':
            masks += [2**i + mask for mask in self.gen_masks(flags[1:], i - 1)]
        if flags[0] in 'N-':
            masks += self.gen_masks(flags[1:], i - 1)
        #print(f"gen_masks({flags}, {i}) -> {masks}")
        return masks

    def decode_mask(self, mask):
        return ''.join([('Y' if 2**i & mask else 'N')
                        for i in range(self.num_columns - 1, -1, -1)])

    def sync_actions(self, actions):
        missing_columns = actions.apply_offset(self.offset, self.num_columns)
        if missing_columns:
            scanner.syntax_error("Missing action for this column",
                                 self.last_rest.lexpos + missing_columns[0],
                                 self.last_rest.lineno)

    def gen_code(self, f, indent=0):
        prefix = indent_str(indent)
        print(f"{prefix}dlt_mask = 0", file=f)
        def gen_bit(i, n):
            print(f"{prefix}if ({n}) dlt_mask |= (2 << {i})", file=f)
        for i, expr in enumerate(self.exprs):
            gen_bit(len(self.exprs) - 1, expr)
        print(f"{prefix}switch (dlt_mask)", "{", file=f)


class Actions(Symtable):
    r'''
    self.blocks is [(rest, [column_number], [statement])]
    '''
    def __init__(self, actions):
        r'self.actions is ((statement, rest), ...)'
        self.compile_actions(actions)

    def compile_actions(self, actions):
        r'''Creates self.blocks.

        Checks for duplicate markers in the same column.
        '''
        if not actions[0][1].value.strip():
            rest = actions[0][1]
            scanner.syntax_error("Missing column marker(s) on first DLT action",
                                 rest.lexpos, rest.lineno)

        # Gather blocks of actions.
        # A block starts with an action that has column markers, and includes
        # all of the following actions without column markers.
        self.blocks = []  # (rest, [column_number], [statement])
        seen = set()
        for statement, rest in actions:
            markers = rest.value.rstrip()
            if markers:
                column_numbers = []
                for i, marker in enumerate(markers):
                    if marker in 'xX':
                        if i in seen:
                            scanner.syntax_error("Duplicate action for column",
                                                 rest.lexpos + i, rest.lineno)
                        column_numbers.append(i)
                    elif marker != ' ':
                        scanner.syntax_error("Column marker must be 'X' or ' '",
                                             rest.lexpos + i, rest.lineno)
                self.blocks.append((rest, column_numbers, [statement]))
            else:
                self.blocks[-1][2].append(statement)

    def apply_offset(self, offset, num_columns):
        r'''Subtracts offset from each column_number in self.blocks.

        Checks that all column_numbers are >= offset and resulting
        column_number is < num_columns.

        Returns an ordered list of missing column_numbers.
        '''
        seen = set()
        for rest, column_numbers, _ in self.blocks:
            for i in range(len(column_numbers)):
                if column_numbers[i] < offset or \
                   column_numbers[i] - offset >= num_columns:
                    scanner.syntax_error("Column marker does not match "
                                         "any column in DLT conditions section",
                                         rest.lexpos + column_numbers[i],
                                         rest.lineno)
                column_numbers[i] -= offset
                seen.add(column_numbers[i])

        # Return missing column_numbers.
        return sorted(frozenset(range(num_columns)) - seen)

    def gen_code(self, f, indent=0):
        prefix = indent_str(indent)
        for _, column_numbers, statements in self.blocks:
            for i in column_numbers:
                print(f"{prefix}case {i}:", file=f)
            prefix = indent_str(indent + 4)
            for s in statements:
                print(f"{prefix}{s};", file=f)
        print(prefix, end='', file=f)
        print("}", file=f)


class DLT(Symtable):
    def __init__(self, conditions, actions):
        self.conditions = conditions
        self.actions = actions
        conditions.sync_actions(actions)

    def gen_code(self, f, indent=0):
        self.conditions.gen_code(f, indent)
        self.actions.gen_code(f, indent)


def indent_str(indent):
    return " "*indent

