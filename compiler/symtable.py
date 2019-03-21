# symtable.py

from collections import OrderedDict

import scanner


def push_namespace(ns):
    scanner.Namespaces.append(ns)
    return entity


def pop_namespace():
    del scanner.Namespaces[-1]


def current_namespace():
    return scanner.Namespaces[-1]


def top_namespace():
    return scanner.Namespaces[0]


class Entity:
    def __init__(self, ident):
        self.name = ident.value
        if scanner.Namespaces:
            scanner.Namespaces[-1].set(ident, self)

    def dump(self, f, indent=0):
        print(f"{indent_str(indent)}{self.__class__.__name__} {self.name}",
              end='', file=f)
        self.dump_details(f)

    def dump_details(self, f):
        print(file=f)


class Variable(Entity):
    def __init__(self, ident, scope='global', type=None):
        Entity.__init__(self, ident)
        self.type = type or ident.type
        self.scope = scope

    def dump_details(self, f):
        print(f"type={self.type} scope={self.scope}", file=f)


class Use(Entity):
    def __init__(self, alias, module_name, arguments):
        Entity.__init__(self, alias)
        self.module_name = module_name
        self.arguments = arguments

    def dump_details(self, f):
        print(f"module_name={self.module_name}", file=f)


class Typedef(Entity):
    def __init__(self, ident, definition):
        Entity.__init__(self, ident)
        self.definition = definition

    def dump_details(self, f):
        print(f"definition={self.definition}", file=f)


class Block:
    def __init__(self, statements):
        self.statements = statements


class Conditions:
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


class Actions:
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


class DLT:
    def __init__(self, conditions, actions):
        self.conditions = conditions
        self.actions = actions
        conditions.sync_actions(actions)


class With_parameters:
    def __init__(self):
        self.current_parameters = self.pos_parameters = []
        self.kw_parameters = OrderedDict()
        self.seen_default = False

    def pos_parameter(self, ident, default=None):
        if default is None and self.seen_default:
            scanner.syntax_error("Required parameter may not follow "
                                 "default parameter",
                                 ident.lexpos,
                                 ident.lineno)
        self.current_parameters.append((ident, default))
        if default is not None:
            self.seen_default = True

    def kw_parameter(self, keyword):
        lname = keyword.value.lower()
        if lname in self.kw_parameters:
            scanner.syntax_error("Duplicate keyword",
                                 keyword.lexpos, keyword.lineno)
        self.current_parameters = self.kw_parameters[lname] = []
        self.seen_default = False

    def dump_details(self, f):
        print(f"pos_params {self.pos_parameters} "
                f"kw_params {self.kw_parameters.items()}",
              file=f)


class Labeled_block(Entity, With_parameters):
    def __init__(self, ident):
        Entity.__init__(self, ident)
        With_parameters.__init__(self)

    def add_first_block(self, block):
        self.block = block


class Namespace(Entity):
    def __init__(self, ident):
        Entity.__init__(self, ident)
        self.names = OrderedDict()
        #print(f"scanner.Namespaces.append({ident})")
        scanner.Namespaces.append(self)

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.name}>"

    def set(self, ident, entity):
        lname = ident.value.lower()
        if lname in self.names:
            scanner.syntax_error("Duplicate name", ident.lexpos, ident.lineno)
        self.names[lname] = entity

    def dump(self, f, indent=0):
        super().dump(f, indent)
        for entity in self.names.values():
            entity.dump(f, indent + 2)


class Dummy_token:
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"<Dummy_token {self.value!r}>"


class Opmode(Namespace):
    def __init__(self):
        Namespace.__init__(self, Dummy_token(scanner.file_basename()))


class Module(Opmode, With_parameters):
    def __init__(self):
        Opmode.__init__(self)
        With_parameters.__init__(self)


class Subroutine(Namespace, With_parameters):
    def __init__(self, ident):
        Namespace.__init__(self, ident)
        With_parameters.__init__(self)

    def add_first_block(self, first_block):
        self.first_block = first_block


class Function(Subroutine):
    def __init__(self, ident):
        Subroutine.__init__(self, ident)
        self.return_type = ident.type

    def set_return_types(self, return_types):
        self.return_types = return_types




def indent_str(indent):
    return " "*indent

