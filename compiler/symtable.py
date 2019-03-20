# symtable.py

from collections import OrderedDict


Symtables = []


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


class Use:
    def __init__(self, module_name, arguments):
        self.module_name = module_name
        self.arguments = arguments


class Typedef:
    def __init__(self, code_type, taking, returning=None):
        self.code_type = code_type
        self.taking = taking
        self.returning = returning


def lookup(name, *, type=Symbol, **kws):
    return Symtables[-1].lookup(name, type=type, **kws)


class Entity:
    def __init__(self, sym):
        self.name = sym
        sym.entity = self
        self.symbols = OrderedDict()

    def lookup(self, name, *, type=Symbol, **kws):
        lname = name.lower()
        if lname in self.symbols:
            return self.symbols[lname]
        sym = type(name, **kws)
        self.symbols[lname] = sym
        return sym


class Opmode(Entity):
    pass


class Entity_with_parameters(Entity):
    def __init__(self, sym):
        super().__init__(sym)
        self.current_parameters = '__pos__'
        self.kw_parameters = {'__pos__': self.current_parameters}

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
            self.lookup(label, type='label',
                        generator=Parameterized_block(params, statements))
        else:
            self.lookup(label, type='label', generator=Block(statements))


class Block:
    def __init__(self, statements):
        self.statements = statements


class Parameterized_block(Block):
    def __init__(self, params, statements):
        super(self).__init__(statements)
        self.parameters = params


class Subroutine(Routine):
    pass


class Function(Routine):
    def __init__(self, sym):
        super().__init__(sym)
        self.return_types = (sym.type,)
