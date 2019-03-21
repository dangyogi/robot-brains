# symtable.py

from collections import OrderedDict


def push(obj):
    Symtables.append(obj)
    return obj


def pop():
    del Symtables[-1]


def current_entity():
    return Symtables[-1]


def find_global(self):
    if self.entity is not None:
        return self.entity.find(self)
    return self


def find_local(self):
    return self


class Symbol:
    entity = None

    def __init__(self, name, **kws):
        self.name = name
        self.find = find_global
        for k, v in kws.items():
            setattr(self, k, v)

    def __repr__(self):
        return f"<Symbol {self.name}>"

    def dump(self, f, indent = 0):
        print(f"{indent_str(indent)}Symbol.{self.name}", end='', file=f)
        if hasattr(self, 'entity'):
            print(f" entity={self.entity}", end='', file=f)
        if hasattr(self, 'type'):
            print(f" type={self.type}", end='', file=f)
        if hasattr(self, 'scope'):
            print(f" scope={self.scope}", end='', file=f)
        print(file=f)
        if self.entity is not None:
            self.entity.dump(f, indent + 2)


class Use:
    def __init__(self, module_name, arguments):
        self.module_name = module_name
        self.arguments = arguments

    def __repr__(self):
        return f"<Use {self.module_name} {self.arguments}>"


class Typedef:
    def __init__(self, code_type, taking, returning=None):
        self.code_type = code_type
        self.taking = taking
        self.returning = returning

    def __repr__(self):
        return f"<Typedef {self.code_type} taking {self.taking} " \
               f"returning {self.returning}>"


def lookup(name, *, sym_class=Symbol, **kws):
    return Symtables[-1].lookup(name, sym_class=sym_class, **kws)


class Entity:
    def __init__(self, sym):
        self.name = sym
        sym.entity = self
        self.symbols = OrderedDict()

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.name}>"

    def lookup(self, name, *, sym_class=Symbol, **kws):
        lname = name.lower()
        if lname in self.symbols:
            return self.symbols[lname]
        sym = sym_class(name, **kws)
        self.symbols[lname] = sym
        return sym

    def dump(self, f, indent = 0):
        print(f"{indent_str(indent)}{self.__class__.__name__}.{self.name}",
              file=f)
        for _, sym in self.symbols.items():
            sym.dump(f, indent + 2)


class Opmode(Entity):
    pass


class Entity_with_parameters(Entity):
    def __init__(self, sym):
        super().__init__(sym)
        self.current_parameters = '__pos__'
        self.kw_parameters = {'__pos__': []}

    def kw_parameter(self, t):
        lkw_name = kw_name.lower()
        if lkw_name in self.kw_parameters:
            syntax_error(t, "Duplicate keyword")
        self.current_parameters = kw_name
        self.kw_parameters[lkw_name] = []

    def pos_parameter(self, symbol, default=None):
        self.kw_parameters[self.current_parameters].append((symbol, default))
        symbol.scope = 'parameter'
        if self.current_parameters == '__pos__':
            self.kw_param_name = None
        else:
            self.kw_param_name = self.current_parameters
        symbol.param_number = len(self.kw_parameters[self.current_parameters])


class Module(Entity_with_parameters):
    pass


class Routine(Entity_with_parameters):
    block = None

    def add_block(self, statements):
        assert self.block is None
        self.block = Block(statements)

    def labeled_block(self, label, params, statements):
        if params:
            self.lookup(label.name, type='label',
                        generator=Parameterized_block(params, statements))
        else:
            self.lookup(label.name, type='label', generator=Block(statements))


class Block:
    def __init__(self, statements):
        self.statements = statements

    def __repr__(self):
        return f"<Block>"


class Parameterized_block(Block):
    def __init__(self, params, statements):
        super(self).__init__(statements)
        self.parameters = params

    def __repr__(self):
        return f"<Parameterized_block {self.params}>"


class Subroutine(Routine):
    pass


class Function(Routine):
    def __init__(self, sym):
        super().__init__(sym)
        self.return_types = (sym.type,)


def indent_str(indent):
    return " "*indent


Symtables = [Entity(Symbol('__file__', type='file'))]
