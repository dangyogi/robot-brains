# symtable.py

from collections import OrderedDict, ChainMap

import scanner


Last_parameter_obj = None


def last_parameter_obj():
    return Last_parameter_obj


def current_namespace():
    return scanner.Namespaces[-1]


def top_namespace():
    return scanner.Namespaces[0]


class Symtable:
    r'''Root of all symtable classes.

    Defines dump.
    '''
    def dump(self, f, indent=0):
        print(f"{indent_str(indent)}{self.__class__.__name__}", end='', file=f)
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
    def __init__(self, ident):
        self.name = ident
        self.set_in_namespace()

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.name.value}>"

    def set_in_namespace(self):
        self.parent_namespace = current_namespace()
        self.parent_namespace.set(self.name, self)

    def dump_details(self, f):
        super().dump_details(f)
        print(f" {self.name.value}", end='', file=f)

    def prepare(self, opmode, module):
        r'''Called once (per module, not instance).
        '''
        pass

    def prepare_step(self, opmode, module, last_label, last_fn_subr):
        r'''Called once (per module, not instance).
        '''
        pass


class Variable(Entity):
    r'''self.set_bit is the bit number in module.vars_set.  Set in prepare.
    '''
    explicit_typedef = False
    dimensions = ()

    def __init__(self, ident, **kws):
        Entity.__init__(self, ident)
        if 'type' in kws:
            self.explicit_typedef = True
            # self.type set in loop below
        else:
            self.type = Builtin_type(ident.type)
        for k, v in kws.items():
            setattr(self, k, v)
        self.parameter_list = []        # list of parameters using this Variable
        assert current_namespace().name != 'builtins' \
            or ident.value != 'p_config'

    def dump_details(self, f):
        super().dump_details(f)
        print(f" type={self.type}", end='', file=f)

    def prepare(self, opmode, module):
        super().prepare(opmode, module)
        if not self.dimensions:
            self.set_bit = module.next_set_bit
            module.next_set_bit += 1
        self.type.prepare(opmode, module)


class Required_parameter(Symtable):
    r'''
    self.param_pos_number assigned in Param_block.add_parameter.
    '''
    def __init__(self, ident):
        self.name = ident
        self.variable = current_namespace().make_variable(ident)
        self.variable.parameter_list.append(self)
        Last_parameter_obj.pos_parameter(self)

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.name.value}>"

    def dump_details(self, f):
        super().dump_details(f)
        print(' ', self.name.value, sep='', end='', file=f)
        if hasattr(self, 'param_pos_number'):
            print(f" param_pos_number={self.param_pos_number}", end='', file=f)

    def prepare(self, opmode, module):
        self.type = self.variable.type


class Optional_parameter(Required_parameter):
    r'''
    self.passed_bit assigned in Param_block.assign_passed_bits.  This is the
    bit number, not the bit mask.
    '''
    pass


class Use(Entity):
    def __init__(self, ident, module_name, arguments):
        Entity.__init__(self, ident)
        self.module_name = module_name
        self.arguments = arguments

    def __repr__(self):
        return f"<Use {self.name.value} {self.module_name.value}>"

    def prepare(self, opmode, module):
        super().prepare(opmode, module)
        self.module = opmode.modules_seen[self.module_name.value]

    def dump_details(self, f):
        super().dump_details(f)
        print(f" module_name={self.module_name}", end='', file=f)


class Typedef(Entity):
    r'''type is either:

       (FUNCTION, taking_opt, returning_opt)
    or (SUBROUTINE, taking_opt)
    or (LABEL, taking_opt)
    or IDENT
    or MODULE
    '''

    is_module = False

    def __init__(self, ident, type):
        Entity.__init__(self, ident)
        self.type = type

    def __repr__(self):
        return f"<Typedef {self.name.value} {self.type}>"

    def dump_details(self, f):
        super().dump_details(f)
        print(f" type={self.type}", end='', file=f)

    def prepare(self, opmode, module):
        super().prepare(opmode, module)
        self.type.prepare(opmode, module)


class Type(Symtable):
    def prepare(self, opmode, module):
        pass


class Builtin_type(Type):
    def __init__(self, name):
        self.name = name  # str

    def __repr__(self):
        return f"<Builtin_type {self.name}>"


class Typename_type(Type):
    def __init__(self, ident):
        self.ident = ident

    def __repr__(self):
        return f"<Typename {self.ident.value}>"

    def prepare(self, opmode, module):
        super().prepare(opmode, module)

        # types can not be passed as module parameters, so this will be the
        # same result for all module instances...
        self.typedef = lookup(self.ident, module, opmode)
        if not isinstance(self.typedef, Typedef):
            scanner.syntax_error("Must be defined as a TYPE, "
                                   f"not a {self.typedef.__class__.__name__}",
                                 self.ident.lexpos, self.ident.lineno,
                                 self.ident.filename)


class Label_type(Type):
    def __init__(self, label_type, taking_blocks, return_types=()):
        # self.label_type is "subroutine", "function" or "label"
        self.label_type = label_type.lower()

        if taking_blocks is None:
            self.required_params = self.optional_params = ()
            self.kw_params = ()
        else:
            assert len(taking_blocks) == 2
            assert len(taking_blocks[0]) == 2
            self.required_params, self.optional_params = taking_blocks[0]
            self.kw_params = taking_blocks[1]
        self.return_types = return_types

    def __repr__(self):
        info = []
        if self.required_params:
            info.append(f"required_params: {self.required_params}")
        if self.optional_params:
            info.append(f"optional_params: {self.optional_params}")
        if self.kw_params:
            info.append(f"kw_params: {self.kw_params}")
        if self.return_types:
            info.append(f"returning: {self.return_types}")
        if info:
            return f"<Label_type {self.label_type} {' '.join(info)}>"
        return f"<Label_type {self.label_type}>"


class Param_block(Symtable):
    def __init__(self, name="__pos__", optional=False):
        self.name = name
        self.required_params = []
        self.optional_params = []
        self.optional = optional

    def dump_details(self, f):
        super().dump_details(f)
        print(' ', self.name, sep='', end='', file=f)
        if self.optional:
            print(" optional", end='', file=f)

    def dump_contents(self, f, indent):
        super().dump_contents(f, indent)
        for p in self.required_params:
            p.dump(f, indent+2)
        for p in self.optional_params:
            p.dump(f, indent+2)

    def add_parameter(self, param):
        if isinstance(param, Optional_parameter):
            param.param_pos_number = \
              len(self.required_params) + len(self.optional_params)
            self.optional_params.append(param)
        else:  # required parameter
            assert not self.optional_params
            param.param_pos_number = len(self.required_params)
            self.required_params.append(param)

    def gen_parameters(self):
        r'''Generates all parameter objects, both required and optional.
        '''
        yield from self.required_params
        yield from self.optional_params

    def lookup_param(self, ident, error_not_found=True):
        r'''Returns None if not found, unless error_not_found.
        '''
        lname = ident.value.lower()
        for p in self.gen_parameters():
            if lname == p.name.value.lower():
                return p
        if error_not_found:
            scanner.syntax_error("Parameter Not Found",
                                 ident.lexpos, ident.lineno, ident.filename)
        return None

    def prepare(self, opmode, module):
        #print("Param_block.prepare")
        for p in self.gen_parameters():
            #print("Param_block.prepare", p)
            p.prepare(opmode, module)

    def assign_passed_bits(self, next_bit_number):
        def get_bit_number():
            nonlocal next_bit_number
            ans = next_bit_number
            next_bit_number += 1
            return ans
        if self.optional:
            self.kw_passed_bit = get_bit_number()
        for p in self.optional_params:
            p.passed_bit = get_bit_number()
        return next_bit_number

    def as_taking_block(self):
        r'Must run prepare first!'
        required_types = tuple(p.type for p in self.required_params)
        optional_types = tuple(p.type for p in self.optional_params)
        if self.name == '__pos__':
            return required_types, optional_types
        return self.name, required_types, optional_types


class Namespace(Symtable):
    r'''Collection of named Entities.
    '''
    def __init__(self, name):
        self.name = name
        self.names = {}
        #print(f"scanner.Namespaces.append({ident})")
        scanner.Namespaces.append(self)

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.name}>"

    def lookup(self, ident, error_not_found=True):
        r'''Look up ident in self.names.

        if not found:
            if error_not_found: generate scanner.syntax_error
            else: return None
        '''
        lname = ident.value.lower()
        #print(self.name, "lookup", lname, tuple(self.names.keys()))
        if lname in self.names:
            return self.names[lname]
        if error_not_found:
            scanner.syntax_error("Undefined name", ident.lexpos, ident.lineno)
        return None

    def set(self, ident, entity):
        r'''Stores entity in self.names under ident.

        Reports duplicate names through scanner.syntax_error.
        '''
        lname = ident.value.lower()
        if lname in self.names:
            print("other entity", self.names[lname])
            scanner.syntax_error("Duplicate definition",
                                 ident.lexpos, ident.lineno)
        self.names[lname] = entity

    def make_variable(self, ident):
        lname = ident.value.lower()
        if lname not in self.names:
            entity = self.names[lname] = Variable(ident)
        else:
            entity = self.names[lname]
            if not isinstance(entity, Variable):
                scanner.syntax_error(
                  f"Duplicate definition with {entity.__class__.__name__} "
                  f"on line {entity.name.lineno}",
                  ident.lexpos, ident.lineno)
        return entity

    def dump_contents(self, f, indent):
        for entity in self.names.values():
            entity.dump(f, indent)


class Opmode(Namespace):
    def __init__(self, modules_seen=None):
        Namespace.__init__(self, scanner.file_basename())
        self.filename = scanner.filename()

        if modules_seen is not None:
            # {module_name: module}
            # ... does not have lowered() keys!
            self.modules_seen = modules_seen
        self.next_set_bit = 0

    def set_in_namespace(self):
        pass

    #def dump_details(self, f):
    #    super().dump_details(f)
    #    print(f" modules_seen: {','.join(self.modules_seen.keys())}",
    #          end='', file=f)

    def dump_contents(self, f, indent):
        super().dump_contents(f, indent)
        if not isinstance(self, Module):
            print(f"{indent_str(indent)}Modules:", file=f)
            indent += 2
            for name, module in self.modules_seen.items():
                print(f"{indent_str(indent)}{name}:", file=f)
                module.dump(f, indent + 2)

    def prepare(self, opmode, module):
        #print("Opmode.prepare", self.__class__.__name__, self.name)
        for obj in self.names.values():
            obj.prepare(opmode, module)
        print(self.name, self.__class__.__name__,
              "number of var-set bits used", self.next_set_bit)


class With_parameters(Symtable):
    def __init__(self):
        global Last_parameter_obj
        self.pos_param_block = None
        self.current_param_block = None
        self.kw_parameters = OrderedDict()      # {lname: Param_block}
        Last_parameter_obj = self

    def pos_parameter(self, param):
        if self.current_param_block is None:
            self.pos_param_block = Param_block()
            self.current_param_block = self.pos_param_block
        self.current_param_block.add_parameter(param)

    def kw_parameter(self, keyword, optional=False):
        lname = keyword.value.lower()
        if lname in self.kw_parameters:
            scanner.syntax_error("Duplicate keyword",
                                 keyword.lexpos, keyword.lineno)
        self.current_param_block = self.kw_parameters[lname] = \
          Param_block(keyword.value, optional)

    def dump_contents(self, f, indent):
        for pb in self.gen_param_blocks():
            pb.dump(f, indent)
        super().dump_contents(f, indent)

    def gen_param_blocks(self):
        if self.pos_param_block is not None:
            yield self.pos_param_block
        yield from self.kw_parameters.values()

    def lookup_param(self, ident, error_not_found=True):
        r'''Lookup parameter name in all Param_blocks.

        Returns None if not found, unless error_not_found.
        '''
        for pb in self.gen_param_blocks():
            param = lookup_param(ident, error_not_found=False)
            if param is not None:
                return param
        if error_not_found:
            scanner.syntax_error("Parameter Not Found",
                                 ident.lexpos, ident.lineno, ident.filename)

    def prepare(self, opmode, module):
        super().prepare(opmode, module)
        for pb in self.gen_param_blocks():
            pb.prepare(opmode, module)
        self.check_for_duplicate_names()
        self.assign_passed_bits()

    def check_for_duplicate_names(self):
        seen = set()
        for pb in self.gen_param_blocks():
            for p in pb.gen_parameters():
                lname = p.name.value.lower()
                if lname in seen:
                    scanner.syntax_error("Duplicate Parameter Name",
                                         p.name.lexpos, p.name.lineno,
                                         p.name.filename)
                seen.add(lname)

    def assign_passed_bits(self):
        next_bit_number = 0
        for pb in self.gen_param_blocks():
            next_bit_number = pb.assign_passed_bits(next_bit_number)
        self.bits_used = next_bit_number

    def as_taking_blocks(self):
        r'Must run prepare first!'
        if self.pos_param_block is None:
            pos_block = (), ()
        else:
            pos_block = self.pos_param_block.as_taking_block()
        return (pos_block,
                tuple(pb.as_taking_block()
                      for pb in self.kw_parameters.values()))


class Module(With_parameters, Opmode):
    r'''self.steps is a tuple of labels, statements and DLTs.
    '''
    def __init__(self):
        Opmode.__init__(self)
        With_parameters.__init__(self)

    def add_steps(self, steps):
        self.steps = steps

    def dump_contents(self, f, indent):
        super().dump_contents(f, indent)
        for step in self.steps:
            step.dump(f, indent)

    def prepare(self, opmode, module):
        super().prepare(opmode, module)
        last_label = None
        last_fn_subr = None
        for step in self.steps:
            if isinstance(step, Subroutine):
                last_fn_subr = step
            if isinstance(step, Label):
                last_label = step
            assert last_label is not None
            step.prepare_step(opmode, module, last_label, last_fn_subr)


class Label(With_parameters, Entity):
    def __init__(self, ident):
        Entity.__init__(self, ident)
        With_parameters.__init__(self)

    def prepare(self, opmode, module):
        super().prepare(opmode, module)
        self.type = Label_type(self.__class__.__name__,
                               self.as_taking_blocks(),
                               self.get_returning_types())

    def get_returning_types(self):
        return ()


class Subroutine(Label):
    struct_name_suffix = "_sub"


class Function(Subroutine):
    struct_name_suffix = "_fn"

    def __init__(self, ident):
        Subroutine.__init__(self, ident)
        self.return_types = Builtin_type(ident.type),

    def set_return_types(self, return_types):
        self.return_types = return_types

    def get_returning_types(self):
        return self.return_types

    def dump_details(self, f):
        super().dump_details(f)
        print(f" return_types {self.return_types}", end='', file=f)


class Native_subroutine(With_parameters, Entity):
    def __init__(self, ident):
        Entity.__init__(self, ident)
        With_parameters.__init__(self)

    def add_lines(self, lines):
        self.lines = lines


class Native_function(Native_subroutine):
    def __init__(self, ident):
        Native_subroutine.__init__(self, ident)
        self.type = Builtin_type(ident.type)

    def set_return_type(self, return_type):
        self.type = return_type


class Statement(Symtable):
    arguments = {
        'continue': (),
        'set': ('lvalue', 'expr'),
        'goto': ('expr', ('*', 'expr')),
        'return': (('*', 'expr'), 'ignore', 'expr'),
    }

    formats = {
        'set': ('{0[0]} = {0[1]}',),
        'goto': ('goto {}',),
        'return': ('return {}',),
    }

    def __init__(self, lineno, *args):
        self.lineno = lineno
        self.args = args

    def dump_details(self, f):
        super().dump_details(f)
        print(f" lineno {self.lineno}", end='', file=f)
        print(f" {self.args}", end='', file=f)

    def prepare_step(self, opmode, module, last_label, last_fn_subr):
        def _prepare(obj):
            if isinstance(obj, Symtable):
                obj.prepare_step(opmode, module, last_label, last_fn_subr)
            elif isinstance(obj, (list, tuple)):
                for x in obj:
                    _prepare(x)
        for arg in self.args:
            _prepare(arg)

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
    # primary arguments [RETURNING_TO primary]
    pass


class Opeq_statement(Statement):
    # primary OPEQ primary
    pass


class Done_statement(Statement):
    # done [with: label]
    pass


class Conditions(Symtable):
    r'''DLT conditions section.

    self.exprs is condition expressions in order from MSB to LSB in the mask.
    self.num_columns is the number of columns in the DLT.
    self.offset is the number of spaces between the '|' and the first column.
    self.column_masks has one list of masks for each column.
    self.dlt_mask is the dlt_mask Token (for future syntax errors).
    '''
    def __init__(self, conditions):
        r'conditions is ((mask, expr), ...)'
        self.compile_conditions(conditions)

    def compile_conditions(self, conditions):
        self.exprs = []  # exprs form bits in mask, ordered MSB to LSB
        self.offset = None
        for dlt_mask, expr in conditions:
            if self.offset is None:
                self.offset = len(dlt_mask.value) - len(dlt_mask.value.lstrip())
                self.num_columns = len(dlt_mask.value.strip())
                columns = [[] for _ in range(self.num_columns)]
            this_offset = len(dlt_mask.value) - len(dlt_mask.value.lstrip())
            if this_offset != self.offset:
                scanner.syntax_error("Condition flags don't start in the same "
                                     "column as previous condition",
                                     dlt_mask.lexpos + this_offset,
                                     dlt_mask.lineno)
            dlt_mask.value = dlt_mask.value.lstrip()
            dlt_mask.lexpos += this_offset
            if len(dlt_mask.value.strip()) != self.num_columns:
                scanner.syntax_error("Must have same number of condition flags "
                                     "as previous condition",
                                     dlt_mask.lexpos, dlt_mask.lineno)
            self.exprs.append(expr)
            for i, flag in enumerate(dlt_mask.value.strip()):
                if flag not in '-yYnN':
                    scanner.syntax_error("Illegal condition flag, "
                                         "expected 'y', 'n', or '-'",
                                         dlt_mask.lexpos + i, dlt_mask.lineno)
                columns[i].append(flag.upper())

        #print("exprs", self.exprs, "offset", self.offset,
        #      "num_columns", self.num_columns, "columns", columns)

        self.dlt_mask = dlt_mask

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
                                         self.dlt_mask.lexpos + i,
                                         self.dlt_mask.lineno)
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
                                 self.dlt_mask.lexpos + self.num_columns,
                                 self.dlt_mask.lineno)

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
                                 self.dlt_mask.lexpos + missing_columns[0],
                                 self.dlt_mask.lineno)

    def prepare_step(self, opmode, module, last_label, last_fn_subr):
        for expr in self.exprs:
            expr.prepare_step(opmode, module, last_label, last_fn_subr)

    def gen_code(self, f, indent=0):
        prefix = indent_str(indent)
        print(f"{prefix}dlt_mask = 0", file=f)
        def gen_bit(i, n):
            print(f"{prefix}if ({n}) dlt_mask |= (2 << {i})", file=f)
        for i, expr in enumerate(self.exprs):
            gen_bit(len(self.exprs) - 1, expr)
        print(f"{prefix}switch (dlt_mask)", "{", file=f)


class DLT_MAP(Symtable):
    r'''self.map is a Token whose value is the "X X..." map.
    '''
    def __init__(self, map):
        self.map = map

    def assign_column_numbers(self, seen):
        r'''Creates self.column_numbers.

        Reports duplicate actions for any column.
        '''
        self.column_numbers = []
        for i, marker in enumerate(self.map.value.lower()):
            if marker == 'x':
                if i in seen:
                    scanner.syntax_error("Duplicate action for column",
                                         self.map.lexpos + i, self.map.lineno)
                self.column_numbers.append(i)

    def apply_offset(self, offset, num_columns):
        r'''Subtracts offset from each column_number in self.column_numbers.

        Checks that all column_numbers are >= offset and resulting
        column_number is < num_columns.

        Returns a list of the resulting column_numbers.
        '''
        for i in range(len(self.column_numbers)):
            if self.column_numbers[i] < offset or \
               self.column_numbers[i] - offset >= num_columns:
                scanner.syntax_error("Column marker does not match "
                                     "any column in DLT conditions section",
                                     self.map.lexpos + self.column_numbers[i],
                                     self.map.lineno)
            self.column_numbers[i] -= offset
        return self.column_numbers

    def dump_details(self, f):
        super().dump_details(f)
        print(f" {self.map.value}", end='', file=f)

    def prepare_step(self, opmode, module, last_label, last_fn_subr):
        pass


class Actions(Symtable):
    def __init__(self, actions):
        r'self.actions is ((dlt_map | statement) ...)'
        self.actions = actions
        seen = set()
        for action in actions:
            if isinstance(action, DLT_MAP):
                action.assign_column_numbers(seen)

    def apply_offset(self, offset, num_columns):
        r'''Subtracts offset from each column_number in self.actions.

        Checks that all column_numbers are >= offset and resulting
        column_number is < num_columns.

        Returns an ordered list of missing column_numbers.
        '''
        seen = set()
        for action in self.actions:
            if isinstance(action, DLT_MAP):
                seen.update(action.apply_offset(offset, num_columns))

        # Return missing column_numbers.
        ans = sorted(frozenset(range(num_columns)) - seen)
        return ans

    def prepare_step(self, opmode, module, last_label, last_fn_subr):
        for action in self.actions:
            action.prepare_step(opmode, module, last_label, last_fn_subr)

    def gen_code(self, f, indent=0):
        # FIX: this is old and obsolete.  Delete???
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
        self.conditions = conditions    # Conditions instance
        self.actions = actions          # Actions instance
        conditions.sync_actions(actions)

    def prepare_step(self, opmode, module, last_label, last_fn_subr):
        self.conditions.prepare_step(opmode, module, last_label, last_fn_subr)
        self.actions.prepare_step(opmode, module, last_label, last_fn_subr)


class Expr(Symtable):
    def prepare_step(self, opmode, module, last_label, last_fn_subr):
        pass


class Literal(Expr):
    def __init__(self, value):
        self.value = value
        if isinstance(value, bool):
            self.type = 'boolean'
        elif isinstance(value, int):
            self.type = 'integer'
        elif isinstance(value, float):
            self.type = 'float'
        elif isinstance(value, str):
            self.type = 'string'

    def __repr__(self):
        return f"<Literal {self.value!r}>"


class Variable_ref(Expr):
    def __init__(self, ident):
        self.ident = ident

    def __repr__(self):
        return f"<Variable_ref {self.ident.value}>"

    def prepare_step(self, opmode, module, last_label, last_fn_subr):
        super().prepare_step(opmode, module, last_label, last_fn_subr)

        # These should be the same for all instances (since the type is
        # defined in the module and can't be passed in as a module parameter).
        self.referent = lookup(self.ident, module, opmode)
        self.type = self.referent.type


class Dot(Expr):
    def __init__(self, expr, ident):
        self.expr = expr
        self.ident = ident

    def __repr__(self):
        return f"<Dot {self.expr} {self.ident.value}>"

    # self.type has to be done for each instance...
    def prepare_step(self, opmode, module, last_label, last_fn_subr):
        super().prepare_step(opmode, module, last_label, last_fn_subr)
        self.expr.prepare_step(opmode, module, last_label, last_fn_subr)
        # FIX: self.expr should evaluate to a module at compile time
        #      then lookup self.ident at compile time


class Subscript(Expr):
    def __init__(self, array_expr, subscript_expr):
        self.array_expr = array_expr
        self.subscript_expr = subscript_expr

    def __repr__(self):
        return f"<Subscript {self.array_expr} {self.subscript_expr}>"

    def prepare_step(self, opmode, module, last_label, last_fn_subr):
        super().prepare_step(opmode, module, last_label, last_fn_subr)
        self.array_expr.prepare_step(opmode, module, last_label, last_fn_subr)
        self.subscript_expr.prepare_step(opmode, module, last_label,
                                         last_fn_subr)
        # self.type has to be done for each instance...
        # self.type = self.array_expr.type


class Got_keyword(Expr):
    r'''self.label and self.bit_number set in prepare.
    '''
    def __init__(self, keyword, label=None, module=False):
        self.keyword = keyword
        self.label = label
        self.module = module
        self.type = 'boolean'

    def __repr__(self):
        return f"<Got_keyword {self.keyword}>"

    def prepare_step(self, opmode, module, last_label, last_fn_subr):
        super().prepare_step(opmode, module, last_label, last_fn_subr)
        if self.module:
            pb = module.kw_parameters.get(self.keyword.value.lower())
        else:
            if self.label is None:
                self.label = last_label
            pb = self.label.kw_parameters.get(self.keyword.value.lower())
        if pb is None:
            scanner.syntax_error("Keyword Not Found",
                                 self.keyword.lexpos, self.keyword.lineno,
                                 self.keyword.filename)
        if not pb.optional:
            scanner.syntax_error("Must be an optional keyword",
                                 self.keyword.lexpos, self.keyword.lineno,
                                 self.keyword.filename)
        self.bit_number = pb.kw_passed_bit


class Got_param(Expr):
    r'''self.label and self.bit_number set in prepare.
    '''
    def __init__(self, ident, label=None, module=False):
        self.ident = ident
        self.label = label
        self.module = module
        self.type = 'boolean'

    def __repr__(self):
        return f"<Got_param {self.ident.value}>"

    def prepare_step(self, opmode, module, last_label, last_fn_subr):
        super().prepare_step(opmode, module, last_label, last_fn_subr)
        if self.module:
            obj = module
        else:
            if self.label is None:
                self.label = last_label
            obj = self.label
        found = False
        for pb in obj.gen_param_blocks():
            param = pb.lookup_param(self.ident, error_not_found=False)
            if param is not None:
                if isinstance(param, Optional_parameter):
                    self.bit_number = param.passed_bit
                    break
                else:
                    scanner.syntax_error("Must be an optional parameter",
                                         self.ident.lexpos,
                                         self.ident.lineno,
                                         self.ident.filename)
        else:
            scanner.syntax_error("Parameter Not Found",
                                 self.ident.lexpos, self.ident.lineno,
                                 self.ident.filename)


class Call_fn(Expr):
    def __init__(self, fn, arguments):
        self.fn = fn
        self.pos_arguments, self.kw_arguments = arguments

    def __repr__(self):
        return f"<Call_fn {self.fn} {self.pos_arguments} {self.kw_arguments}>"

    def prepare_step(self, opmode, module, last_label, last_fn_subr):
        # self.type has to be done for each instance...
        super().prepare_step(opmode, module, last_label, last_fn_subr)
        print("Call_fn.prepare_step", self.fn)
        self.fn.prepare_step(opmode, module, last_label, last_fn_subr)
        for expr in self.pos_arguments:
            expr.prepare_step(opmode, module, last_label, last_fn_subr)
        for _, pos_arguments in self.kw_arguments:
            for expr in pos_arguments:
                expr.prepare_step(opmode, module, last_label, last_fn_subr)


class Unary_expr(Expr):
    def __init__(self, operator, expr):
        self.operator = operator
        self.expr = expr

    def __repr__(self):
        return f"<Unary_expr {self.operator} {self.expr}>"

    def prepare_step(self, opmode, module, last_label):
        super().prepare_step(opmode, module, last_label)
        self.expr.prepare_step(opmode, module, last_label)
        # FIX: type check and self.type has to be done for each instance...


class Binary_expr(Expr):
    def __init__(self, expr1, operator, expr2):
        self.expr1 = expr1
        self.operator = operator
        self.expr2 = expr2

    def __repr__(self):
        return f"<Binary_expr {self.expr1} {self.operator} {self.expr2}>"

    def prepare_step(self, opmode, module, last_label, last_fn_subr):
        super().prepare_step(opmode, module, last_label, last_fn_subr)
        self.expr1.prepare_step(opmode, module, last_label, last_fn_subr)
        self.expr2.prepare_step(opmode, module, last_label, last_fn_subr)
        # FIX: type check and self.type has to be done for each instance...


class Return_label(Expr):
    def __init__(self, lexpos, lineno, label=None):
        #print("Return_label", lexpos, lineno, label)
        self.lexpos = lexpos
        self.lineno = lineno
        self.filename = scanner.lexer.filename
        self.label = label      # IDENT

    def prepare_step(self, opmode, module, last_label, last_fn_subr):
        super().prepare_step(opmode, module, last_label, last_fn_subr)
        if self.label is None:
            self.label = last_label
        # FIX: lookup self.label


def indent_str(indent):
    return " "*indent


def lookup(ident, module, opmode):
    #print("lookup", ident, module, opmode)
    obj = module.lookup(ident, error_not_found=False)
    if obj is not None:
        return obj
    obj = opmode.lookup(ident, error_not_found=False)
    if obj is not None:
        return obj
    obj = opmode.modules_seen['builtins'].lookup(ident, error_not_found=False)
    if obj is not None:
        return obj
    scanner.syntax_error("Not found",
                         ident.lexpos, ident.lineno, ident.filename)
