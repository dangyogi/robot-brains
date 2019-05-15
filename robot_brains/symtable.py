# symtable.py

from collections import OrderedDict, defaultdict
from copy import deepcopy
from math import isclose
from itertools import tee, zip_longest, chain

from robot_brains import scanner


Last_parameter_obj = None


def pairwise(iterable, fillvalue=None):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip_longest(a, b, fillvalue=fillvalue)


def aeq(x, y):
    return isclose(x, y, rel_tol=0, abs_tol=1e-6)


def last_parameter_obj():
    return Last_parameter_obj


def current_namespace():
    return scanner.Namespaces[-1]


def top_namespace():
    return scanner.Namespaces[0]


class Hook_base:
    r'''Implements hook capability.

    - Allows multiple fns to be registered per cls, hook_name.
    - Calls all fns found (in reverse __mro__ order) when the hook is
      triggered.
    '''
    pre_hooks = defaultdict(list)     # {(cls, hook_name): [fn]}
    post_hooks = defaultdict(list)    # {(cls, hook_name): [fn]}

    @classmethod
    def add_pre_hook(cls, hook_name, fn):
        Hook_base.pre_hooks[cls, hook_name].append(fn)

    @classmethod
    def add_post_hook(cls, hook_name, fn):
        Hook_base.post_hooks[cls, hook_name].append(fn)

    @classmethod
    def as_pre_hook(cls, hook_name):
        r'Function decorator'
        def decorator(fn):
            cls.add_pre_hook(hook_name, fn)
            return fn
        return decorator

    @classmethod
    def as_post_hook(cls, hook_name):
        r'Function decorator'
        def decorator(fn):
            cls.add_post_hook(hook_name, fn)
            return fn
        return decorator

    def run_pre_hooks(self, hook_name, *args, **kws):
        r'''Passes self to the hook fns, in addition to *args and **kws.
        '''
        for c in reversed(self.__class__.__mro__):
            for fn in Hook_base.pre_hooks.get((c, hook_name), ()):
                fn(self, *args, **kws)

    def run_post_hooks(self, hook_name, *args, **kws):
        r'''Passes self to the hook fns, in addition to *args and **kws.
        '''
        for c in reversed(self.__class__.__mro__):
            for fn in Hook_base.post_hooks.get((c, hook_name), ()):
                fn(self, *args, **kws)


class Symtable(Hook_base):
    r'''Root of all symtable classes.

    Defines dump, prepare and prepare_step.
    '''
    prepared = False

    def dump(self, f, indent=0):
        print(f"{indent_str(indent)}{self.__class__.__name__}", end='', file=f)
        self.dump_details(f)
        print(file=f)
        self.dump_contents(f, indent + 2)

    def dump_details(self, f):
        pass

    def dump_contents(self, f, indent):
        pass

    def prepare(self, module):
        r'''Called on each instance.
        '''
        if not self.prepared:
            self.prepared = True
            self.run_pre_hooks("prepare", module)
            self.do_prepare(module)
            self.run_post_hooks("prepare", module)

    def do_prepare(self, module):
        r'''Override this in the subclasses.
        '''
        pass


class Entity(Symtable):
    r'''Named entities within a namespace.
    '''
    def __init__(self, ident, namespace=None):
        self.name = ident
        self.set_in_namespace(namespace)

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.name.value}>"

    def set_in_namespace(self, namespace=None):
        if namespace is None:
            self.parent_namespace = current_namespace()
        else:
            self.parent_namespace = namespace
        self.parent_namespace.set(self.name, self)

    def dump_details(self, f):
        super().dump_details(f)
        print(f" {self.name.value}", end='', file=f)


class Variable(Entity):
    r'''self.set_bit is the bit number in module.vars_set.  Set in prepare.
    '''
    explicit_typedef = False
    dimensions = ()   # (int, ...)
    immediate = False
    constant = False
    prep_statements = ()
    vars_used = frozenset()

    def __init__(self, ident, namespace=None, **kws):
        Entity.__init__(self, ident, namespace)
        self.lexpos = ident.lexpos
        self.lineno = ident.lineno
        self.filename = ident.filename
        self.type = Builtin_type(ident.type)
        for k, v in kws.items():
            setattr(self, k, v)
        self.parameter_list = []        # list of parameters using this Variable

    def dump_details(self, f):
        super().dump_details(f)
        print(f" type={self.type}", end='', file=f)

    def do_prepare(self, module):
        super().do_prepare(module)
        if not self.dimensions:
            self.set_bit = module.next_set_bit
            self.vars_used = frozenset((self,))
            module.next_set_bit += 1
        self.type.prepare(module)

    def set_module(self, module):
        self.immediate = True
        assert module.immediate
        self.value = module.value

    def get_step(self):
        return self

    def is_numeric(self):
        return self.type.is_numeric()

    def is_integer(self):
        return self.type.is_integer()

    def is_float(self):
        return self.type.is_float()

    def is_string(self):
        return self.type.is_string()

    def is_boolean(self):
        return self.type.is_boolean()

    def is_module(self):
        return self.type.is_module()


class Required_parameter(Symtable):
    r'''
    self.passed_bit is bit number in label.params_passed
    self.param_pos_number assigned in Param_block.add_parameter.
    '''
    def __init__(self, ident, label=None):
        self.name = ident
        if label is None:
            self.variable = current_namespace().make_variable(ident)
            Last_parameter_obj.pos_parameter(self)
        else:
            self.variable = label.parent_namespace.make_variable(ident)
            label.pos_parameter(self)
        self.variable.parameter_list.append(self)

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.name.value}>"

    def dump_details(self, f):
        super().dump_details(f)
        print(' ', self.name.value, sep='', end='', file=f)
        if hasattr(self, 'param_pos_number'):
            print(f" param_pos_number={self.param_pos_number}", end='', file=f)

    def do_prepare(self, module):
        super().do_prepare(module)
        self.type = self.variable.type


class Optional_parameter(Required_parameter):
    r'''
    self.passed_bit assigned in Param_block.assign_passed_bits.  This is the
    bit number, not the bit mask.

    self.passed is set to True/False at compile time for module parameters.
    '''
    pass


class Use(Entity):
    def __init__(self, ident, module_name, pos_arguments, kw_arguments):
        Entity.__init__(self, ident)
        self.module_name = module_name
        self.pos_arguments = pos_arguments  # (primary, ...)

        # ((KEYWORD_TOKEN, (primary, ...)), ...)
        self.kw_arguments = kw_arguments

    def __repr__(self):
        return f"<Use {self.name.value} {self.module_name.value}>"

    def copy_module(self, parent_module):
        reference_module = Opmode_module.modules_seen[self.module_name.value]
        self.module = deepcopy(reference_module)
        reference_module.instance_count += 1
        self.module.instance_number = reference_module.instance_count
        self.module.parent_module = parent_module
        self.module.link_uses()

    def dump_details(self, f):
        super().dump_details(f)
        print(f" module_name={self.module_name}", end='', file=f)

    def prepare_used_module(self, module):
        r'''
        Broken out so that builtins and opmode get prepared before used modules
        so that their Entities are initialized before being referenced in used
        modules.
        '''
        #print("Use.prepare_used_module", module, self.module_name)
        self.run_pre_hooks("prepare_used_module", module)
        for expr in self.pos_arguments:
            expr.prepare_step(module, None, None)
        for _, exprs in self.kw_arguments:
            for expr in exprs:
                expr.prepare_step(module, None, None)
        #print("Use.prepare_used_module", module, self.module_name,
        #      "done with arguments")

        #print(module, "use passing constant module parameters")

        self.module.prepare_module()

        self.module.params_type.satisfied_by_arguments(
                                  Error_reporter(self.name),
                                  self.pos_arguments, self.kw_arguments)
        pass_args(self.module, self.pos_arguments, self.kw_arguments,
                  Use_param_compiler())
        self.run_post_hooks("prepare_used_module", module)
        #print("Use.prepare_used_module", module, self.module_name, "done")

    @property
    def type(self):
        return self.module.type

    def get_step(self):
        return self.module


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

    def cmp_params(self):
        return (self.module.filename, self.name.value.lower())

    def dump_details(self, f):
        super().dump_details(f)
        print(f" type={self.type}", end='', file=f)

    def do_prepare(self, module):
        super().do_prepare(module)
        self.module = module
        self.type.prepare(module)


class Type(Symtable):
    any_type = False

    def __eq__(self, b):
        return type(self) == type(b) and self.cmp_params() == b.cmp_params()

    def __hash__(self):
        return hash((self.__class__, self.cmp_params()))

    def get_type(self):
        return self

    def is_numeric(self):
        return False

    def is_integer(self):
        return False

    def is_float(self):
        return False

    def is_string(self):
        return False

    def is_boolean(self):
        return False

    def is_module(self):
        return False

    def can_take_type(self, reporter, arg_type):
        r'''Check and report any type mismatches.

        If there are any mismatches, the syntax error is reported against
        `arg_type` if `report_arg_type` is True, else against self.
        '''
        my_t = self.get_type()
        arg_t = arg_type.get_type()
        if not my_t.any_type and not arg_t.any_type and my_t is not arg_t:
            if type(my_t) != type(arg_t):
                reporter.report(f"Incompatible types, {my_t} and {arg_t}")
            my_t._can_take_type(reporter, arg_t)


class Any_type(Type):
    any_type = True

    def __str__(self):
        return "Any_type"

    def is_numeric(self):
        return True

    def is_integer(self):
        return True

    def is_float(self):
        return True

    def is_string(self):
        return True

    def is_boolean(self):
        return True

    def is_module(self):
        return True


class Builtin_type(Type):
    def __init__(self, name):
        self.name = name.lower()  # str

    def __repr__(self):
        return f"<Builtin_type {self.name}>"

    def __str__(self):
        return self.name.upper()

    def cmp_params(self):
        return self.name.lower()

    def is_numeric(self):
        return self.name in ('float', 'integer')

    def is_integer(self):
        return self.name == 'integer'

    def is_float(self):
        return self.name == 'float'

    def is_string(self):
        return self.name == 'string'

    def is_boolean(self):
        return self.name == 'boolean'

    def is_module(self):
        return self.name == 'module'

    def _can_take_type(self, reporter, arg_type):
        if not (self.name == arg_type.name or \
                self.name == 'float' and arg_type.name == 'integer'):
            reporter.report(f"Incompatible types, {self} and {arg_type}")


class Typename_type(Type):
    def __init__(self, path):
        self.path = path
        self.lexpos, self.lineno, self.filename = \
          self.path[-1].lexpos, self.path[-1].lineno, self.path[-1].filename

    def __repr__(self):
        return f"<Typename {self}>"

    def __str__(self):
        return '.'.join(p.value for p in self.path)

    def cmp_params(self):
        return self.typedef.cmp_params()

    def get_type(self):
        r'Only valid after prepared.'
        return self.typedef.type.get_type()

    def do_prepare(self, module):
        super().do_prepare(module)
        for p in self.path[:-1]:
            module = lookup(p, module)
            if not isinstance(module, Use):
                scanner.syntax_error("Must be defined as a MODULE, "
                                       f"not a {module.__class__.__name__}",
                                     p.lexpos, p.lineno, p.filename)
            module = module.module
        self.typedef = lookup(self.path[-1], module)
        if not isinstance(self.typedef, Typedef):
            scanner.syntax_error("Must be defined as a TYPE, "
                                   f"not a {self.typedef.__class__.__name__}",
                                 self.path[-1].lexpos, self.path[-1].lineno,
                                 self.path[-1].filename)

    def is_numeric(self):
        return self.typedef.type.is_numeric()

    def is_integer(self):
        return self.typedef.type.is_integer()

    def is_float(self):
        return self.typedef.type.is_float()

    def is_string(self):
        return self.typedef.type.is_string()

    def is_boolean(self):
        return self.typedef.type.is_boolean()

    def is_module(self):
        return self.typedef.type.is_module()


class Label_type(Type):
    r'''
    self.label_type is "subroutine", "function" or "label"
    self.required_params is (type, ...)
    self.optional_params is (type, ...)
    self.kw_params is {KEYWORD_TOKEN: (kw_optional, (type, ...), (type, ...))}
    self.return_types is ((req, opt), ((keyword_token, (req, opt)), ...))
    '''
    def __init__(self, label_type, taking_blocks, return_types=None):
        self.label_type = label_type.lower()

        if taking_blocks is None:
            self.required_params = self.optional_params = ()
            self.kw_params = {}
        else:
            assert len(taking_blocks) == 2
            assert len(taking_blocks[0]) == 2
            self.required_params, self.optional_params = taking_blocks[0]
            self.kw_params = {}
            for keyword, (req, opt) in taking_blocks[1]:
                if keyword.value[0] == '?':
                    self.kw_params[keyword] = (True, req, opt)
                else:
                    self.kw_params[keyword] = (False, req, opt)
        if self.label_type == 'function':
            self.return_types = return_types
            self.return_label_type = Label_type('label', return_types)
        else:
            assert return_types is None
            if self.label_type == 'subroutine':
                self.return_label_type = Label_type('label', None)

    def __repr__(self):
        info = []
        if self.required_params:
            info.append(f"required_params: {self.required_params}")
        if self.optional_params:
            info.append(f"optional_params: {self.optional_params}")
        if self.kw_params:
            info.append(f"kw_params: {self.kw_params}")
        if hasattr(self, 'return_types'):
            info.append(f"returning: {self.return_types}")
        if info:
            return f"<Label_type {self.label_type} {' '.join(info)}>"
        return f"<Label_type {self.label_type}>"

    def __str__(self):
        return self.label_type.upper()

    def cmp_params(self):
        ans = [self.label_type, self.required_params, self.optional_params,
               tuple(sorted((kw.value.lower(), rest)
                            for kw, rest in self.kw_params.items()))]
        if self.label_type == 'function':
            ans.append((self.return_types[0],
                        tuple(sorted((kw.value.lower(), rest)
                                     for kw, rest in self.return_types[1]))))
        return tuple(ans)

    def do_prepare(self, module):
        super().do_prepare(module)
        def prepare_type(t):
            t.prepare(module)
            return t
        for t in self.required_params:
            t.prepare(module)
        for t in self.optional_params:
            t.prepare(module)
        for (_, req, opt) in self.kw_params.values():
            for t in req:
                t.prepare(module)
            for t in opt:
                t.prepare(module)
        if hasattr(self, 'return_label_type'):
            self.return_label_type.prepare(module)
        # self.return_types have already been passed to the Label_type for
        # return_label_type and have been prepared by that Label_type.

    def ok_as_return_for(self, reporter, fn_subr_type):
        r'''
        Check and report type mismatch between myself and fn return type.

        Assumes I'm at fault if there are any mismatches.
        '''
        fn_ret_t = fn_subr_type.get_type().return_label_type.get_type()
        fn_ret_t.can_take_type(reporter.return_type(), self)

    def _can_take_type(self, reporter, arg_type):
        r'''
        I am a LABEL variable, and arg_type is an actual LABEL.  Can arg_type
        safely be assigned to this variable?

        I.e., do I provide all of args that arg_type requires, and can
        arg_type accept all of the args that I might provide?

        Finally, if I expect a return, can my return accept the arg_type
        return?  I.e., can the arg_type return label variable take my return
        label?

        Calls reporter.report to report any errors.  Expects it to raise an
        exception.
        '''

        if (self.label_type == 'label') != (arg_type.label_type == 'label'):
            # Must both be labels, or neither labels; otherwise one expects to
            # return and the other doesn't.
            reporter.report(
              f"Incompatable labels, {self.label_type.upper()} and "
                f"{arg_type.label_type.upper()}")

        arg_type.satisfied_by(reporter, self.required_params,
                              self.optional_params, self.kw_params)

        # For subroutines and functions, check the return_label_type
        if self.label_type != 'label':  # then arg_type is not 'label' either
            # Can the arg_type take my return label to return to?
            arg_type.return_label_type.can_take_type(reporter.return_type(),
                                                     self.return_label_type)

    def satisfied_by(self, reporter, req, opt=(), kw_params={}):
        r'''Do req, opt, kw_params satisfy my parameter needs?
        
        Without anything extra that I don't know about?

        kw_params is {KEYWORD_TOKEN: (kw_optional, (type, ...), (type, ...))}

        Calls reporter.report to report errors.  Expects it to raise an
        exception.

        Does not check return types.
        '''

        def check_pos_params(reporter, receiving_req, receiving_opt,
                             sending_req, sending_opt):
            # receiving is always from self, sending from req/opt/kw_params.
            if len(sending_req) + len(sending_opt) > \
               len(receiving_req) + len(receiving_opt):
                # receiving type must be able to take max number of params
                max_rec_len = len(receiving_req) + len(receiving_opt)
                max_send_len = len(sending_req) + len(sending_opt)
                reporter.report(
                  f"Mismatch in max number of parameters, {max_rec_len} and "
                    f"{max_send_len}")
            if len(sending_req) < len(receiving_req):
                # receiving type must be able to take min number of params
                reporter.report(
                  f"Mismatch in min number of parameters, {len(receiving_req)} "
                    f"and {len(sending_req)}")
            for i, (sending_param, receiving_param) \
             in enumerate(zip(chain(sending_req, sending_opt),
                              chain(receiving_req, receiving_opt)),
                          1):
                receiving_param.get_type().can_take_type(
                                             reporter.param_number(i),
                                             sending_param)

        # Check positonal parameters
        check_pos_params(reporter.positional(),
                         self.required_params, self.optional_params, req, opt)

        # Check kw_params
        sending_req_kws = set()
        for keyword, (kw_optional, req, opt) in kw_params.items():
            if keyword not in self.kw_params:
                reporter.report(f"Unknown keyword {keyword.value}")
            my_kw_optional, my_req, my_opt = self.kw_params[keyword]
            if kw_optional:
                if not my_kw_optional:
                    reporter.report("Required/Optional mismatch for "
                                      f"{keyword.value}")
            else:
                sending_req_kws.add(keyword)
            check_pos_params(reporter.keyword(keyword),
                             my_req, my_opt, req, opt)

        # Do I have any required kws that may not be provided?
        for my_keyword, (my_kw_optional, _, _) in self.kw_params.items():
            if not my_kw_optional and my_keyword not in sending_req_kws:
                reporter.report("Required/Optional mismatch for "
                                  f"{my_keyword.value}")

    def satisfied_by_arguments(self, reporter, pos_arguments, kw_arguments=()):
        r'''Checks if arguments are OK for me.

        kw_arguments is ((KEYWORD_TOKEN, (primary, ...)), ...)

        If there are any mismatches, a syntax_error is reported on the
        arguments.

        Does not check return type.
        '''
        self.satisfied_by(reporter,
                          tuple(e.type for e in pos_arguments),
                          (),
                          {keyword: (False, 
                                     tuple(e.type for e in pos_arguments),
                                     ())
                           for keyword, pos_arguments in kw_arguments})


class Param_block(Symtable):
    r'''
    self.kw_passed_bit is bit number of kw_passed bit in label.params_passed
    self.passed is set to True/False at compile time for module parameters.
    '''
    def __init__(self, name=None, optional=False):
        self.name = name  # token
        self.required_params = []
        self.optional_params = []
        self.optional = optional

    def dump_details(self, f):
        super().dump_details(f)
        if self.name is not None:
            print(' ', self.name.value, sep='', end='', file=f)
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

    def do_prepare(self, module):
        super().do_prepare(module)
        #print("Param_block.do_prepare")
        for p in self.gen_parameters():
            #print("Param_block.do_prepare", p)
            p.prepare(module)

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
        if self.name is None:
            return required_types, optional_types
        return self.name, (required_types, optional_types)


class Namespace(Symtable):
    r'''Collection of named Entities.
    '''
    def __init__(self, name):
        self.name = name
        self.names = OrderedDict()
        #print(f"scanner.Namespaces.append({ident})")
        scanner.Namespaces.append(self)
        self.entities_prepared = False

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.name}>"

    def filter(self, cls):
        return (entity
                for entity in self.names.values()
                 if isinstance(entity, cls))

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
            scanner.syntax_error("Undefined name", ident.lexpos, ident.lineno,
                                 ident.filename)
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
            # Variable automatically places itself in self.names
            entity = Variable(ident)
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

    def prepare_entities(self):
        if not self.entities_prepared:
            self.entities_prepared = True
            self.run_pre_hooks("prepare_entities")
            self.do_prepare_entities()
            self.run_post_hooks("prepare_entities")

    def do_prepare_entities(self):
        for obj in self.names.values():
            obj.prepare(self)


class Opmode(Namespace):
    def __init__(self, modules_seen=None):
        Namespace.__init__(self, scanner.file_basename())
        self.filename = scanner.filename()

        if modules_seen is not None:
            # {module_name: module}
            # ... does not have lowered() keys!
            self.modules_seen = modules_seen
        self.next_set_bit = 0
        self.register()

    def register(self):
        global Opmode_module
        Opmode_module = self

    def set_in_namespace(self, namespace=None):
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

    def setup(self):
        print("setup: linking uses")
        builtins = self.modules_seen['builtins']
        builtins.instance_number = 1
        builtins.link_uses()
        self.link_uses()
        print("setup: doing prepare")
        builtins.prepare_module()
        builtins.prepare_module_steps()
        self.prepare_module()
        self.prepare_module_steps()
        print("setup: done")

    def link_uses(self):
        #print(self.name, self.__class__.__name__, "link_uses")
        self.run_pre_hooks("link_uses")
        for use in self.filter(Use):
            #print(self.name, "link_uses doing", use)
            use.copy_module(self)
        self.run_post_hooks("link_uses")

    def prepare_module(self):
        r'''Called on each module instance.
        '''
        #print(self.name, self.__class__.__name__, "prepare_module")
        self.run_pre_hooks("prepare_module")
        self.do_prepare_module()
        self.run_post_hooks("prepare_module")

    def do_prepare_module(self):
        #print("Opmode.do_prepare_module", self.__class__.__name__, self.name)
        self.prepare_entities()
        self.prepare(self)   # declared in With_parameters
        for use in self.filter(Use):
            #print(self.name, "link_uses doing", use)
            use.prepare_used_module(self)
        print(self.name, self.__class__.__name__,
              "number of var-set bits used", self.next_set_bit)

    def prepare_module_steps(self):
        for use in self.filter(Use):
            use.module.prepare_module_steps()


class With_parameters(Symtable):
    def __init__(self):
        global Last_parameter_obj
        self.pos_param_block = None
        self.current_param_block = None
        self.kw_parameters = OrderedDict()      # {keyword_token: Param_block}
        Last_parameter_obj = self

    def pos_parameter(self, param):
        if self.current_param_block is None:
            self.pos_param_block = Param_block()
            self.current_param_block = self.pos_param_block
        self.current_param_block.add_parameter(param)

    def kw_parameter(self, keyword, optional=False):
        r'`keyword` is a Token.'
        if keyword in self.kw_parameters:
            scanner.syntax_error("Duplicate keyword",
                                 keyword.lexpos, keyword.lineno)
        self.current_param_block = self.kw_parameters[keyword] = \
          Param_block(keyword, optional)

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
            param = pb.lookup_param(ident, error_not_found=False)
            if param is not None:
                return param
        if error_not_found:
            scanner.syntax_error("Parameter Not Found",
                                 ident.lexpos, ident.lineno, ident.filename)

    def do_prepare(self, module):
        super().do_prepare(module)
        for pb in self.gen_param_blocks():
            pb.prepare(module)
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
        next_bit_number = 1   # Leave room for RUNNING bit
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
    type = Builtin_type("module")
    immediate = True
    parent_module = None

    def __init__(self):
        Opmode.__init__(self)
        With_parameters.__init__(self)
        self.instance_count = 0  # only used in the reference modules created
                                 # directly by the parser

    def register(self):
        pass

    @property
    def value(self):
        return self

    def add_steps(self, steps):
        self.steps = steps

    def dump_contents(self, f, indent):
        super().dump_contents(f, indent)
        for step in self.steps:
            step.dump(f, indent)

    def do_prepare_module(self):
        r'''Only called on builtins module.
        '''
        super().do_prepare_module()
        # Make all module parameters constant variables.
        for pb in self.gen_param_blocks():
            for p in pb.gen_parameters():
                p.variable.constant = True

        self.params_type = Label_type('label', self.as_taking_blocks())
        self.params_type.prepare(self)

    def prepare_module_steps(self):
        print(self, "prepare_module_steps")
        last_label = None
        last_fn_subr = None
        for step, next_step in pairwise(self.steps):
            if isinstance(step, Subroutine):
                last_fn_subr = step
                last_label = step
            elif isinstance(step, Label) and not step.hidden:
                last_label = step
            assert last_label is not None
            last_label.reset()  # makes all temps available again
            step.prepare_step(self, last_label, last_fn_subr)
            if next_step is None or isinstance(next_step, Label):
                if not step.is_final():
                    scanner.syntax_error("Statement must not fall-through",
                                         step.lexpos, step.lineno,
                                         step.filename)
            else:
                if step.is_final():
                    scanner.syntax_error("Statement must fall-through",
                                         step.lexpos, step.lineno,
                                         step.filename)
        super().prepare_module_steps()


class Step(Symtable):
    step_prepared = False

    def prepare_step(self, module, last_label, last_fn_subr):
        r'''Called once per instance.
        '''
        if not self.step_prepared:
            self.step_prepared = True
            self.run_pre_hooks("prepare_step", module, last_label, last_fn_subr)
            self.do_prepare_step(module, last_label, last_fn_subr)
            self.run_post_hooks("prepare_step", module, last_label,
                                last_fn_subr)

    def do_prepare_step(self, module, last_label, last_fn_subr):
        r'''Override this in the subclasses.
        '''
        pass

    def get_step(self):
        return self


class Label(With_parameters, Step, Entity):
    immediate = True
    prep_statements = ()
    vars_used = frozenset()
    precedence = 100
    assoc = None

    def __init__(self, ident, namespace=None, hidden=False):
        Entity.__init__(self, ident, namespace)
        With_parameters.__init__(self)
        self.hidden = hidden
        self.temps = defaultdict(set)   # {type: {var}}
        self.last_temp_number = 0       # to make temp variable names unique
        self.last_label_number = 0      # to make new label names unique
        self.needs_dlt_mask = False

    def new_label(self, module):
        self.last_label_number += 1
        ident = scanner.Token.dummy(
                      f"__{self.name.value}__dummy_{self.last_label_number}")
        ans = Label(ident, namespace=module, hidden=True)
        return ans

    def new_subr_ret_label(self, module):
        r'Create, prepare and return a new dummy subroutine return label.'
        ans = self.new_label(module)
        ans.prepare(module)
        return ans

    def new_fn_ret_label(self, module, var_to_set):
        r'Create, prepare and return a new dummy function return label.'
        ans = self.new_label(module)
        param = Required_parameter(var_to_set.name, ans)
        ans.prepare(module)
        return ans

    def do_prepare(self, module):
        #print(self.__class__.__name__, "do_prepare", self.name)
        super().do_prepare(module)
        self.type = Label_type(self.__class__.__name__, self.as_taking_blocks(),
                               self.get_return_types())
        self.type.prepare(module)

    def get_return_types(self):
        return None

    @property
    def value(self):
        return self

    def is_final(self):
        return False

    def reset(self):
        self.temps_taken = set()  # {var}

    def get_temp(self, type):
        r'Returns prepared Variable of `type`.'
        available_vars = self.temps[type.get_type()] - self.temps_taken
        if available_vars:
            ans = available_vars.pop()
        else:
            self.last_temp_number += 1
            ans = self.create_variable(
                         f"__{self.name.value}__temp_{self.last_temp_number}",
                         type)
            ans.prepare(self.parent_namespace)
            self.temps[type.get_type()].add(ans)
        self.temps_taken.add(ans)
        return ans

    def create_variable(self, name, type):
        ans = Variable(
                scanner.Token.dummy(name),
                self.parent_namespace,
                type=type)
        ans.prepare(self.parent_namespace)
        return ans

    def need_dlt_mask(self):
        self.needs_dlt_mask = True


class Subroutine(Label):
    pass


class Function(Subroutine):
    def __init__(self, ident):
        Subroutine.__init__(self, ident)
        self.return_types = (((Builtin_type(ident.type),), ()), ())

    def set_return_types(self, return_types):
        r'self.return_types is ((req, opt), ((keyword_token, (req, opt)), ...))'
        self.return_types = return_types

    def get_return_types(self):
        return self.return_types

    def dump_details(self, f):
        super().dump_details(f)
        print(f" return_types {self.return_types}", end='', file=f)


class Statement(Step):
    vars_used = frozenset()
    prep__statements = ()
    post_statements = ()

    def __init__(self, lexpos, lineno, *args):
        self.lexpos = lexpos
        self.lineno = lineno
        self.filename = scanner.filename()
        self.args = args

    def is_final(self):
        return False

    def is_continue(self):
        return False

    def dump_details(self, f):
        super().dump_details(f)
        print(f" lineno {self.lineno}", end='', file=f)
        print(f" {self.args}", end='', file=f)

    def do_prepare_step(self, module, last_label, last_fn_subr):
        super().do_prepare_step(module, last_label, last_fn_subr)
        assert last_label is not None
        self.containing_label = last_label
        prep_statements = []
        self.vars_used = set()
        def _prepare(obj):
            if isinstance(obj, Step):
                obj.prepare_step(module, last_label, last_fn_subr)
                prep_statements.extend(obj.prep_statements)
                self.vars_used.update(obj.vars_used)
            elif isinstance(obj, (list, tuple)):
                for x in obj:
                    _prepare(x)
        _prepare(self.args)
        self.prep_statements = tuple(prep_statements)


class Continue(Statement):
    def is_final(self):
        return True

    def is_continue(self):
        return True


class Set(Statement):
    r'''
    SET lvalue primary
    '''
    def __init__(self, lexpos, lineno, *args):
        Statement.__init__(self, lexpos, lineno, *args)
        self.lvalue = self.args[0]
        self.lvalue.set_as_lvalue()
        self.primary = self.args[1]

    def do_prepare_step(self, module, last_label, last_fn_subr):
        super().do_prepare_step(module, last_label, last_fn_subr)
        self.lvalue.type.get_type().can_take_type(Error_reporter(self.primary),
                                                  self.primary.type)


class Native:
    def __init__(self, native_elements):
        self.native_elements = native_elements  # tuple of Expr|str

    def do_prepare_step(self, module, last_label, last_fn_subr):
        super().do_prepare_step(module, last_label, last_fn_subr)
        prep_statements = []
        self.vars_used = set()
        segments = []
        for element in self.native_elements:
            if isinstance(element, Expr):
                element.prepare_step(module, last_label, last_fn_subr)
                self.vars_used.update(element.vars_used)
                prep_statements.extend(element.prep_statements)
                segments.append(element.get_step().code)
            else:
                assert isinstance(element, str)
                segments.append(element)
        self.prep_statements = tuple(prep_statements)
        self.code = ''.join(segments)


class Native_statement(Native, Statement):
    def __init__(self, lexpos, lineno, native_elements):
        Statement.__init__(self, lexpos, lineno)
        Native.__init__(self, native_elements)

    def is_final(self):
        for native_final in ('exit(', 'return ', 'return(', 'goto ',
                             'break;', 'continue;'):
            if self.native_elements[0].startswith(native_final):
                return True
        return False


class Statement_with_arguments(Statement):
    def do_prepare_step(self, module, last_label, last_fn_subr):
        super().do_prepare_step(module, last_label, last_fn_subr)
        self.do_pre_arg_check_prepare(module, last_label, last_fn_subr)
        #print("label_type", repr(self.label_type), self.arguments[0],
        #      self.arguments[1])
        self.label_type.satisfied_by_arguments(self.error_reporter,
                                               self.arguments[0],
                                               self.arguments[1])

    def do_pre_arg_check_prepare(self, module, last_label, last_fn_subr):
        pass


class Goto(Statement_with_arguments):
    r'''
    GOTO primary arguments
    '''
    def __init__(self, lexpos, lineno, *args):
        Statement.__init__(self, lexpos, lineno, *args)
        self.primary = self.args[0]
        self.arguments = self.args[1]
        self.error_reporter = Error_reporter(self.primary)

    def is_final(self):
        return True

    def do_pre_arg_check_prepare(self, module, last_label, last_fn_subr):
        super().do_pre_arg_check_prepare(module, last_label, last_fn_subr)
        self.label_type = self.primary.type.get_type()


class Return(Statement_with_arguments):
    r'''
    RETURN arguments from_opt [TO primary]
    '''
    def __init__(self, lexpos, lineno, *args):
        Statement.__init__(self, lexpos, lineno, *args)
        self.arguments = self.args[0]
        self.from_opt = self.args[1]    # done processing
        self.to_opt = self.args[2]      # where the arguments are delivered
        self.error_reporter = Error_reporter(self)

    def is_final(self):
        return True

    def do_pre_arg_check_prepare(self, module, last_label, last_fn_subr):
        super().do_pre_arg_check_prepare(module, last_label, last_fn_subr)
        if self.from_opt is None:
            assert last_fn_subr is not None
            self.from_opt = last_fn_subr
        if self.to_opt is None:
            self.to_opt = self.from_opt
            self.dest_label = f"({self.to_opt.get_step().code})->return_label"
            self.label_type = self.to_opt.get_step().type.return_label_type
        else:
            self.dest_label = self.to_opt.get_step().code
            self.label_type = self.to_opt.type.get_type()


class Call_statement(Statement_with_arguments):
    # primary arguments [RETURNING_TO primary]
    #
    # arguments is ((primary, ...), kw_arguments)
    #
    # kw_arguments is ((KEYWORD_TOKEN, (primary, ...)), ...)
    def __init__(self, lexpos, lineno, *args):
        Statement.__init__(self, lexpos, lineno, *args)
        self.primary = self.args[0]
        self.arguments = self.args[1]
        self.returning_to = self.args[2]
        self.final = self.returning_to is not None
        self.error_reporter = Error_reporter(self.primary)

    def is_final(self):
        return self.final

    def do_pre_arg_check_prepare(self, module, last_label, last_fn_subr):
        super().do_pre_arg_check_prepare(module, last_label, last_fn_subr)
        fn = self.primary.get_step()

        # This is always verified to be a Label_type
        fn_type = fn.type.get_type()

        self.label_type = fn_type

        # set self.returning_to
        if self.returning_to is None:
            new_label = last_label.new_subr_ret_label(module)
            self.returning_to = new_label
            self.post_statements = (new_label.decl_code,)

        ret_to_t = self.returning_to.type.get_type()
        if not isinstance(ret_to_t, Label_type) or \
           ret_to_t.label_type != 'label':
            scanner.syntax_error("Must be a LABEL",
                                 self.returning_to.lexpos,
                                 self.returning_to.lineno,
                                 self.returning_to.filename)

        if not ret_to_t.required_params and \
           not ret_to_t.optional_params and \
           not ret_to_t.kw_params:
            # return label does not accept any parameters
            if not isinstance(fn_type, Label_type) or \
               fn_type.label_type != 'subroutine':
                scanner.syntax_error("Must be a SUBROUTINE",
                                     self.primary.lexpos,
                                     self.primary.lineno,
                                     self.primary.filename)
        else:
            # return label does accept parameters
            if not ret_to_t.required_params and \
               all(opt for opt, _, _ in ret_to_t.kw_params.values()):
                # return label does not require any parameters
                if not isinstance(fn_type, Label_type) or \
                   fn_type.label_type == 'label':
                    scanner.syntax_error("Must be a SUBROUTINE or FUNCTION",
                                         self.primary.lexpos,
                                         self.primary.lineno,
                                         self.primary.filename)
            else:
                # return label requires some parameters
                if not isinstance(fn_type, Label_type) or \
                   fn_type.label_type != 'function':
                    scanner.syntax_error("Must be a FUNCTION",
                                         self.primary.lexpos,
                                         self.primary.lineno,
                                         self.primary.filename)
            ret_to_t.ok_as_return_for(self.error_reporter, fn_type)

        fn_type.satisfied_by_arguments(self.error_reporter,
                                       self.arguments[0], self.arguments[1])


class Opeq_statement(Statement):
    # lvalue OPEQ primary
    def __init__(self, lexpos, lineno, *args):
        Statement.__init__(self, lexpos, lineno, *args)
        self.lvalue = self.args[0]
        self.lvalue.set_as_lvalue()
        self.operator = self.args[1]
        self.expr = self.args[2]

    def is_final(self):
        return False

    def do_prepare_step(self, module, last_label, last_fn_subr):
        super().do_prepare_step(module, last_label, last_fn_subr)
        if self.operator[0] == '%':
            if not self.lvalue.is_integer():
                scanner.syntax_error("Must be an integer type",
                                     self.lvalue.lexpos, self.lvalue.lineno,
                                     self.lvalue.filename)
        elif self.operator[0] == '^':
            if not self.lvalue.is_float():
                scanner.syntax_error("Must be an float type",
                                     self.lvalue.lexpos, self.lvalue.lineno,
                                     self.lvalue.filename)
        else:
            if not self.lvalue.is_numeric():
                scanner.syntax_error("Must be a numeric type",
                                     self.lvalue.lexpos, self.lvalue.lineno,
                                     self.lvalue.filename)
        if self.lvalue.is_integer():
            if not self.expr.is_integer():
                scanner.syntax_error("Must be an integer type",
                                     self.expr.lexpos, self.expr.lineno,
                                     self.expr.filename)
        else:
            if not self.expr.is_numeric():
                scanner.syntax_error("Must be a numeric type",
                                     self.expr.lexpos, self.expr.lineno,
                                     self.expr.filename)


class Done_statement(Statement):
    # done [with: label]
    def __init__(self, lexpos, lineno, *args):
        Statement.__init__(self, lexpos, lineno, *args)
        self.ident = self.args[0]

    def do_prepare_step(self, module, last_label, last_fn_subr):
        super().do_prepare_step(module, last_label, last_fn_subr)
        assert last_label is not None
        self.containing_label = last_label
        if self.ident is not None:
            self.label = lookup(self.ident, module)
            if not isinstance(self.label, Subroutine):
                scanner.syntax_error("Must be a SUBROUTINE or FUNCTION",
                                     self.ident.lexpos, self.ident.lineno,
                                     self.ident.filename)
        else:
            assert last_fn_subr is not None
            self.label = last_fn_subr


class Conditions(Step):
    r'''DLT conditions section.

    Mask bits are assigned from bottom up, i.e., the last expr in the
    Conditions block is bit 0 (0x1).

    self.exprs is condition expressions in order from MSB to LSB in the mask.
    self.num_columns is the number of columns in the DLT.
    self.offset is the number of spaces between the '|' and the first column.
    self.column_masks has one list of masks for each column.
    self.dlt_mask is the dlt_mask Token (for future syntax errors).
    '''
    def __init__(self, lineno, conditions):
        r'conditions is ((mask, expr), ...)'
        self.lineno = lineno
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
                                     dlt_mask.lineno,
                                     dlt_mask.filename)
            dlt_mask.value = dlt_mask.value.lstrip()
            dlt_mask.lexpos += this_offset
            if len(dlt_mask.value.strip()) != self.num_columns:
                scanner.syntax_error("Must have same number of condition flags "
                                     "as previous condition",
                                     dlt_mask.lexpos, dlt_mask.lineno,
                                     dlt_mask.filename)
            self.exprs.append(expr)
            for i, flag in enumerate(dlt_mask.value.strip()):
                if flag not in '-yYnN':
                    scanner.syntax_error("Illegal condition flag, "
                                         "expected 'y', 'n', or '-'",
                                         dlt_mask.lexpos + i, dlt_mask.lineno,
                                         dlt_mask.filename)
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
                                         self.dlt_mask.lineno,
                                         self.dlt_mask.filename)
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
                                 self.dlt_mask.lineno,
                                 self.dlt_mask.filename)

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
        missing_columns = actions.apply_offset(self)
        if missing_columns:
            scanner.syntax_error("Missing action for this column",
                                 self.dlt_mask.lexpos + missing_columns[0],
                                 self.dlt_mask.lineno,
                                 self.dlt_mask.filename)

    def do_prepare_step(self, module, last_label, last_fn_subr):
        super().do_prepare_step(module, last_label, last_fn_subr)
        assert last_label is not None
        self.containing_label = last_label
        last_label.need_dlt_mask()
        prep_statements = []
        self.vars_used = set()
        for expr in self.exprs:
            expr.prepare_step(module, last_label, last_fn_subr)
            if not expr.is_boolean():
                scanner.syntax_error("Must be boolean expression",
                                     expr.lexpos, expr.lineno, expr.filename)
            self.vars_used.update(expr.vars_used)
            prep_statements.extend(expr.prep_statements)
        self.prep_statements = tuple(prep_statements)


class DLT_MAP(Step):
    r'''self.map is a Token whose value is the "X X..." map.
    self.column_numbers is a list of column numbers (leftmost column position
    starting at 0).
    self.masks is a list of the matching masks.
    '''
    def __init__(self, map):
        self.map = map

    def is_final(self):
        return False

    def is_continue(self):
        return False

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
                seen.add(i)

    def apply_offset(self, conditions):
        r'''Subtracts offset from each column_number in self.column_numbers.

        Checks that all column_numbers are >= offset and resulting
        column_number is < num_columns.

        Returns a list of the resulting column_numbers.
        '''
        offset = conditions.offset
        num_columns = conditions.num_columns
        for i in range(len(self.column_numbers)):
            if self.column_numbers[i] < offset or \
               self.column_numbers[i] - offset >= num_columns:
                scanner.syntax_error("Column marker does not match "
                                     "any column in DLT conditions section",
                                     self.map.lexpos + self.column_numbers[i],
                                     self.map.lineno)
            self.column_numbers[i] -= offset
        self.masks = sorted(chain.from_iterable(
                              conditions.column_masks[column_number]
                              for column_number in self.column_numbers))
        return self.column_numbers

    def dump_details(self, f):
        super().dump_details(f)
        print(f" {self.map.value}", end='', file=f)


class Actions(Step):
    def __init__(self, actions):
        r'self.actions is ((dlt_map | statement) ...)'
        self.actions = actions
        seen = set()
        self.final = True
        for action, next_action in pairwise(actions):
            if isinstance(action, DLT_MAP):
                action.assign_column_numbers(seen)
            elif action.is_continue():
                self.final = False
            if next_action is not None and not isinstance(next_action, DLT_MAP):
                if action.is_final():
                    scanner.syntax_error("Statement must fall-through",
                                         action.lexpos, action.lineno,
                                         action.filename)
        if not actions[-1].is_final():
            self.final = False

    def is_final(self):
        return self.final

    def apply_offset(self, conditions):
        r'''Subtracts offset from each column_number in self.actions.

        Checks that all column_numbers are >= offset and resulting
        column_number is < num_columns.

        Returns an ordered list of missing column_numbers.
        '''
        seen = set()
        for action in self.actions:
            if isinstance(action, DLT_MAP):
                seen.update(action.apply_offset(conditions))

        # Return missing column_numbers.
        ans = sorted(frozenset(range(conditions.num_columns)) - seen)
        return ans

    def do_prepare_step(self, module, last_label, last_fn_subr):
        super().do_prepare_step(module, last_label, last_fn_subr)
        for action in self.actions:
            action.prepare_step(module, last_label, last_fn_subr)


class DLT(Step):
    def __init__(self, conditions, actions):
        self.conditions = conditions    # Conditions instance
        self.actions = actions          # Actions instance
        conditions.sync_actions(actions)

    def do_prepare_step(self, module, last_label, last_fn_subr):
        super().do_prepare_step(module, last_label, last_fn_subr)
        self.conditions.prepare_step(module, last_label, last_fn_subr)
        self.actions.prepare_step(module, last_label, last_fn_subr)

    def is_final(self):
        return self.actions.is_final()


class Expr(Step):
    immediate = False   # True iff value known at compile time
    vars_used = frozenset()
    prep_statements = ()

    def __init__(self, lexpos, lineno):
        self.lexpos = lexpos
        self.lineno = lineno
        self.filename = scanner.filename()
        self.prelude_steps = []

    def is_numeric(self):
        return self.type.is_numeric()

    def is_integer(self):
        return self.type.is_integer()

    def is_float(self):
        return self.type.is_float()

    def is_string(self):
        return self.type.is_string()

    def is_boolean(self):
        return self.type.is_boolean()

    def is_module(self):
        return self.type.is_module()


class Literal(Expr):
    immediate = True
    precedence = 100
    assoc = None

    def __init__(self, lexpos, lineno, value):
        Expr.__init__(self, lexpos, lineno)
        self.value = value
        if isinstance(value, bool):
            self.type = Builtin_type('boolean')
        elif isinstance(value, int):
            self.type = Builtin_type('integer')
        elif isinstance(value, float):
            self.type = Builtin_type('float')
        elif isinstance(value, str):
            self.type = Builtin_type('string')

    def __repr__(self):
        return f"<Literal {self.value!r}>"


class Reference(Expr):
    r'''An IDENT that references either a Variable or Label.

    Type IDENTs are converted to Typename_types rather than References.
    '''
    def __init__(self, ident):
        Expr.__init__(self, ident.lexpos, ident.lineno)
        self.ident = ident
        self.as_lvalue = False

    def __repr__(self):
        return f"<Reference {self.ident.value}>"

    def set_as_lvalue(self):
        current_namespace().make_variable(self.ident)
        self.as_lvalue = True

    def get_step(self):
        return self.referent.get_step()

    def do_prepare_step(self, module, last_label, last_fn_subr):
        super().do_prepare_step(module, last_label, last_fn_subr)

        # These should be the same for all instances (since the type is
        # defined in the module and can't be passed in as a module parameter).
        self.referent = lookup(self.ident, module)
        self.type = self.referent.type
        #print("Reference", self.ident, "found", self.referent)
        if isinstance(self.referent.get_step(), (Variable, Label)):
            self.prep_statements = self.referent.get_step().prep_statements
            if self.as_lvalue:
                self.var_set = self.referent.get_step()
                if not isinstance(self.var_set, Variable):
                    scanner.syntax_error("Variable expected",
                                         self.ident.lexpos, self.ident.lineno,
                                         self.ident.filename)
            else:
                self.vars_used = self.referent.get_step().vars_used
            self.precedence = self.referent.get_step().precedence
            self.assoc = self.referent.get_step().assoc
            self.code = self.referent.get_step().code


class Dot(Expr):
    precedence = 100
    assoc = None

    def __init__(self, expr, ident):
        Expr.__init__(self, expr.lexpos, expr.lineno)
        self.expr = expr
        self.ident = ident
        self.as_lvalue = False

    def __repr__(self):
        return f"<Dot {self.expr} {self.ident.value}>"

    def set_as_lvalue(self):
        #current_namespace().make_variable(self.ident)
        self.as_lvalue = True

    # self.type has to be done for each instance...
    def do_prepare_step(self, module, last_label, last_fn_subr):
        super().do_prepare_step(module, last_label, last_fn_subr)
        self.expr.prepare_step(module, last_label, last_fn_subr)
        if not self.expr.get_step().immediate or \
           not isinstance(self.expr.get_step().value, Module):
            scanner.syntax_error("Module expected",
                                 self.expr.lexpos, self.expr.lineno,
                                 self.expr.filename)
        self.referent = self.expr.get_step().value.lookup(self.ident)
        self.type = self.referent.type
        step = self.referent.get_step()
        if self.as_lvalue:
            if not isinstance(step, Variable):
                scanner.syntax_error("Variable expected",
                                     self.ident.lexpos, self.ident.lineno,
                                     self.ident.filename)
            self.var_set = step
        else:
            if isinstance(step, Variable):
                self.vars_used = frozenset((step,))
        if step.immediate:
            self.immediate = True
            self.value = step.value

    @property
    def code(self):
        return self.referent.code


class Subscript(Expr):
    var_set = None

    def __init__(self, array_expr, subscript_exprs):
        Expr.__init__(self, array_expr.lexpos, array_expr.lineno)
        self.array_expr = array_expr
        self.subscript_exprs = subscript_exprs

    def __repr__(self):
        return f"<Subscript {self.array_expr} {self.subscript_exprs}>"

    def set_as_lvalue(self):
        pass

    def do_prepare_step(self, module, last_label, last_fn_subr):
        super().do_prepare_step(module, last_label, last_fn_subr)
        assert last_label is not None
        self.containing_label = last_label
        prep_statements = []
        self.vars_used = set()
        self.array_expr.prepare_step(module, last_label, last_fn_subr)
        prep_statements.extend(self.array_expr.get_step().prep_statements)
        self.vars_used.update(self.array_expr.get_step().vars_used)
        if not isinstance(self.array_expr.get_step(), Variable):
            scanner.syntax_error("Subscripts may only apply to Variables",
                                 self.array_expr.lexpos, self.array_expr.lineno,
                                 self.array_expr.filename)
        for subscript in self.subscript_exprs:
            subscript.prepare_step(module, last_label, last_fn_subr)
            prep_statements.extend(subscript.get_step().prep_statements)
            self.vars_used.update(subscript.get_step().vars_used)
        dims = len(self.array_expr.get_step().dimensions)
        if dims < len(self.subscript_exprs):
            scanner.syntax_error("Too many subscripts",
                                 self.subscript_exprs[dims].lexpos,
                                 self.subscript_exprs[dims].lineno,
                                 self.subscript_exprs[dims].filename)
        if dims > len(self.subscript_exprs):
            scanner.syntax_error("Not enough subscripts",
                                 self.subscript_exprs[-1].lexpos,
                                 self.subscript_exprs[-1].lineno,
                                 self.subscript_exprs[-1].filename)
        self.type = self.array_expr.type
        self.prep_statements = tuple(prep_statements)


class Got_keyword(Expr):
    r'''self.label and self.bit_number set in prepare.
    '''
    type = Builtin_type("boolean")

    def __init__(self, lexpos, lineno, keyword, label=None, module=False):
        Expr.__init__(self, lexpos, lineno)
        self.keyword = keyword  # Token
        self.label = label
        self.module = module

    def __repr__(self):
        return f"<Got_keyword {self.keyword.value}>"

    def do_prepare_step(self, module, last_label, last_fn_subr):
        super().do_prepare_step(module, last_label, last_fn_subr)
        if self.module:
            pb = module.kw_parameters.get(self.keyword)
        else:
            if self.label is None:
                self.label = last_label
            pb = self.label.kw_parameters.get(self.keyword)
        if pb is None:
            scanner.syntax_error("Keyword Not Found",
                                 self.keyword.lexpos, self.keyword.lineno,
                                 self.keyword.filename)
        if not pb.optional:
            scanner.syntax_error("Must be an optional keyword",
                                 self.keyword.lexpos, self.keyword.lineno,
                                 self.keyword.filename)
        self.bit_number = pb.kw_passed_bit
        if self.module:
            self.immediate = True
            self.value = pb.passed


class Got_param(Expr):
    r'''self.label and self.bit_number set in prepare.
    '''
    type = Builtin_type("boolean")

    def __init__(self, lexpos, lineno, ident, label=None, module=False):
        Expr.__init__(self, lexpos, lineno)
        self.ident = ident
        self.label = label
        self.module = module

    def __repr__(self):
        return f"<Got_param {self.ident.value}>"

    def do_prepare_step(self, module, last_label, last_fn_subr):
        super().do_prepare_step(module, last_label, last_fn_subr)
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
                    if self.module:
                        self.immediate = True
                        self.value = param.passed
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
    precedence = 100
    assoc = None

    def __init__(self, fn, pos_arguments, kw_arguments):
        Expr.__init__(self, fn.lexpos, fn.lineno)
        self.primary = fn
        self.pos_arguments = pos_arguments  # (primary, ...)

        # ((KEYWORD_TOKEN, (primary, ...)), ...)
        self.kw_arguments = kw_arguments 

    def __repr__(self):
        return f"<Call_fn {self.primary} {self.pos_arguments} " \
                f"{self.kw_arguments}>"

    def do_prepare_step(self, module, last_label, last_fn_subr):
        # self.type has to be done for each instance...
        super().do_prepare_step(module, last_label, last_fn_subr)
        assert last_label is not None
        self.containing_label = last_label
        self.primary.prepare_step(module, last_label, last_fn_subr)
        prep_statements = list(self.primary.prep_statements).copy()
        self.vars_used = set(self.primary.vars_used).copy()
        fn_type = self.primary.type.get_type()
        if not isinstance(fn_type, Label_type) or \
           fn_type.label_type != 'function' or \
           len(fn_type.return_types[0][0]) != 1 or \
           fn_type.return_types[0][1] or \
           fn_type.return_types[1]:
            scanner.syntax_error("Must be a FUNCTION returning a single value",
                                 self.primary.lexpos, self.primary.lineno,
                                 self.primary.filename)
        for expr in self.pos_arguments:
            expr.prepare_step(module, last_label, last_fn_subr)
            prep_statements.extend(expr.prep_statements)
            self.vars_used.update(expr.vars_used)
        for _, pos_arguments in self.kw_arguments:
            for expr in pos_arguments:
                expr.prepare_step(module, last_label, last_fn_subr)
                prep_statements.extend(expr.prep_statements)
                self.vars_used.update(expr.vars_used)
        fn_type.satisfied_by_arguments(Error_reporter(self.primary),
                                       self.pos_arguments, self.kw_arguments)
        self.type = fn_type.return_types[0][0][0]
        temp_var = last_label.get_temp(self.type)
        self.code = temp_var.code
        self.ret_label = last_label.new_fn_ret_label(module, temp_var)
        self.prep_statements = tuple(prep_statements)


class Unary_expr(Expr):
    def __init__(self, lexpos, lineno, operator, expr):
        Expr.__init__(self, lexpos, lineno)
        self.operator = operator.lower()
        self.expr = expr

    def __repr__(self):
        return f"<Unary_expr {self.operator} {self.expr}>"

    def do_prepare_step(self, module, last_label, last_fn_subr):
        super().do_prepare_step(module, last_label, last_fn_subr)
        self.expr.prepare_step(module, last_label, last_fn_subr)
        if self.operator in ('-', 'abs'):
            if not self.expr.is_numeric():
                scanner.syntax_error("Must be a numeric type",
                                     self.expr.lexpos, self.expr.lineno,
                                     self.expr.filename)
            self.type = self.expr.type
            if self.expr.get_step().immediate:
                self.immediate = True
                if self.operator == '-':
                    self.value = -self.expr.get_step().value
                else:
                    self.value = abs(self.expr.get_step().value)
        elif self.operator == 'not':
            if not self.expr.is_boolean():
                scanner.syntax_error("Must be boolean",
                                     self.expr.lexpos, self.expr.lineno,
                                     self.expr.filename)
            self.type = Builtin_type("boolean")
            if self.expr.get_step().immediate:
                self.immediate = True
                self.value = not self.expr.get_step().value
        else:
            raise AssertionError(f"Unknown operator {self.operator}")

    @property
    def prep_statements(self):
        return self.expr.get_step().prep_statements

    @property
    def vars_used(self):
        return self.expr.get_step().vars_used


class Binary_expr(Expr):
    operator_map = {  # {operator: (result_type, arg_type, immediate_fn)}
        '^': ('float', 'number', lambda x, y: x**y),
        '*': ('numeric', 'number', lambda x, y: x * y),
        '/': ('float', 'number', lambda x, y: x / y),
        '//': ('integer', 'integer', lambda x, y: x // y),
        '%': ('integer', 'integer', lambda x, y: x % y),
        '+': ('numeric', 'number', lambda x, y: x + y),
        '-': ('numeric', 'number', lambda x, y: x - y),
        '<': ('boolean', 'number or string', lambda x, y: x < y),
        '<=': ('boolean', 'number or string', lambda x, y: x <= y),
        '<~=': ('boolean', 'float', lambda x, y: x < y or aeq(x, y)),
        '>': ('boolean', 'number or string', lambda x, y: x > y),
        '>=': ('boolean', 'number or string', lambda x, y: x >= y),
        '>~=': ('boolean', 'float', lambda x, y: x > y or aeq(x, y)),
        '==': ('boolean', 'number or string', lambda x, y: x == y),
        '~=': ('boolean', 'float', lambda x, y: aeq(x, y)),
        '!=': ('boolean', 'number or string', lambda x, y: x != y),
        '<>': ('boolean', 'number or string', lambda x, y: x != y),
        '!~=': ('boolean', 'float', lambda x, y: not aeq(x, y)),
        '<~>': ('boolean', 'float', lambda x, y: not aeq(x, y)),
    }

    def __init__(self, expr1, operator, expr2):
        Expr.__init__(self, expr1.lexpos, expr1.lineno)
        self.expr1 = expr1
        self.operator = operator
        self.expr2 = expr2

    def __repr__(self):
        return f"<Binary_expr {self.expr1} {self.operator} {self.expr2}>"

    def do_prepare_step(self, module, last_label, last_fn_subr):
        super().do_prepare_step(module, last_label, last_fn_subr)
        self.expr1.prepare_step(module, last_label, last_fn_subr)
        self.expr2.prepare_step(module, last_label, last_fn_subr)

        result_type, arg_type, immed_fn = self.operator_map[self.operator]

        if arg_type == 'integer':
            if not self.expr1.is_integer():
                scanner.syntax_error(
                  f"Expected INTEGER type, got {self.expr1.get_step().type}",
                  self.expr1.lexpos, self.expr1.lineno, self.expr1.filename)
            if not self.expr2.is_integer():
                scanner.syntax_error(
                  f"Expected INTEGER type, got {self.expr2.get_step().type}",
                  self.expr2.lexpos, self.expr2.lineno, self.expr2.filename)
        elif arg_type == 'float':    # at least one has to be float
            if not self.expr1.is_numeric():
                scanner.syntax_error(
                  f"Expected number type, got {self.expr1.get_step().type}",
                  self.expr1.lexpos, self.expr1.lineno, self.expr1.filename)
            if not self.expr2.is_numeric():
                scanner.syntax_error(
                  f"Expected number type, got {self.expr2.get_step().type}",
                  self.expr2.lexpos, self.expr2.lineno, self.expr2.filename)
            if not self.expr1.is_float() and not self.expr2.is_float():
                scanner.syntax_error(
                  f"Expected FLOAT type, got {self.expr1.get_step().type}",
                                     self.expr1.lexpos, self.expr1.lineno,
                                     self.expr1.filename)
        elif arg_type == 'number':
            if not self.expr1.is_numeric():
                scanner.syntax_error(
                  f"Expected number type, got {self.expr1.get_step().type}",
                  self.expr1.lexpos, self.expr1.lineno, self.expr1.filename)
            if not self.expr2.is_numeric():
                scanner.syntax_error(
                  f"Expected number type, got {self.expr2.get_step().type}",
                  self.expr2.lexpos, self.expr2.lineno, self.expr2.filename)
        elif arg_type == 'number or string':
            if not self.expr1.is_numeric() and not self.expr1.is_string():
                scanner.syntax_error(
                  f"Expected {arg_type} type, got {self.expr1.get_step().type}",
                  self.expr1.lexpos, self.expr1.lineno, self.expr1.filename)
            if not self.expr2.is_numeric() and not self.expr2.is_string():
                scanner.syntax_error(
                  f"Expected {arg_type} type, got {self.expr2.get_step().type}",
                  self.expr2.lexpos, self.expr2.lineno, self.expr2.filename)
            if not self.expr1.type.get_type().any_type and \
               (self.expr1.is_numeric() and not self.expr2.is_numeric() or \
                self.expr1.is_string() and not self.expr2.is_string()):
                scanner.syntax_error(
                  f"{self.operator!r}: expression types don't match",
                  self.expr2.lexpos, self.expr2.lineno, self.expr2.filename)
        else:
            raise AssertionError(
                    f"Unknown arg_type: {arg_type} for {self.operator}")
        if result_type == 'numeric':
            self.type = \
              Builtin_type('integer'
                           if self.expr1.is_integer() and \
                              self.expr2.is_integer()
                           else 'float')
        else:
            self.type = Builtin_type(result_type)
        if self.expr1.get_step().immediate and self.expr2.get_step().immediate:
            self.immediate = True
            self.value = immed_fn(self.expr1.get_step().value,
                                  self.expr2.get_step().value)

    @property
    def prep_statements(self):
        return self.expr1.get_step().prep_statements \
             + self.expr2.get_step().prep_statements

    @property
    def vars_used(self):
        return self.expr1.get_step().vars_used.union(
                 self.expr2.get_step().vars_used)


class Return_label(Expr):
    def __init__(self, lexpos, lineno, label=None):
        #print("Return_label", lexpos, lineno, label)
        Expr.__init__(self, lexpos, lineno)
        self.label = label      # IDENT

    def do_prepare_step(self, module, last_label, last_fn_subr):
        super().do_prepare_step(module, last_label, last_fn_subr)
        if self.label is None:
            if last_fn_subr is None:
                scanner.syntax_error("Must follow a SUBROUTINE or FUNCTION",
                                     self.lexpos, self.lineno, self.filename)
            self.label = last_fn_subr
        else:
            self.label = module.lookup(self.label)
            if not isinstance(self.label, Subroutine):
                scanner.syntax_error("Must be a SUBROUTINE or FUNCTION",
                                     self.label.lexpos, self.label.lineno,
                                     self.label.filename)
        self.type = self.label.type.return_label_type
        #print("Return_label: label", self.label.name.value)
        #print("Return_label: label_type", self.type.label_type)
        #print("Return_label: required_params", self.type.required_params)
        #print("Return_label: optional_params", self.type.optional_params)
        #print("Return_label: kw_params", self.type.kw_params)


class Native_expr(Native, Expr):
    precedence = 100
    assoc = None
    type = Any_type()

    def __init__(self, lexpos, lineno, native_elements):
        Expr.__init__(self, lexpos, lineno)
        Native.__init__(self, native_elements)

    def do_prepare_step(self, module, last_label, last_fn_subr):
        super().do_prepare_step(module, last_label, last_fn_subr)
        self.code = f"({self.code})"


def indent_str(indent):
    return " "*indent


def lookup(ident, module, error_not_found=True):
    #print("lookup", ident, module)
    obj = module.lookup(ident, error_not_found=False)
    if obj is not None:
        return obj
    obj = Opmode_module.lookup(ident, error_not_found=False)
    if obj is not None:
        return obj
    obj = Opmode_module.modules_seen['builtins'] \
                       .lookup(ident, error_not_found=False)
    if obj is not None:
        return obj
    if error_not_found:
        scanner.syntax_error("Not found",
                             ident.lexpos, ident.lineno, ident.filename)
    return None


# FIX: Move this back to Use (and cook Use_param_compiler into it)?
def pass_args(with_args, pos_args, kw_args, param_compiler):
    r'''
    param_compiler must have the following methods:

        - set_param_block_passed(param_block, passed)
        - set_param_passed(param, passed)
        - pair_param(param_block, param, expr)
    '''
    def pass_pos_args(pb, args):
        if pb is None:
            return

        param_compiler.set_param_block_passed(pb, True)

        for p, a in zip_longest(pb.gen_parameters(), args):
            if a is None:
                param_compiler.set_param_passed(p, False)
            else:
                param_compiler.set_param_passed(p, True)
                param_compiler.pair_param(pb, p, a)

    pass_pos_args(with_args.pos_param_block, pos_args)

    kws_passed = set()
    for keyword, args in kw_args:
        pass_pos_args(with_args.kw_parameters[keyword], args)
        kws_passed.add(keyword)

    for keyword, pb in with_args.kw_parameters.items():
        if keyword not in kws_passed:
            param_compiler.set_param_block_passed(pb, False)


class Use_param_compiler:
    def set_param_block_passed(self, param_block, passed):
        param_block.passed = passed

    def set_param_passed(self, param, passed):
        param.passed = passed

    def pair_param(self, param_block, param, expr):
        if expr.get_step().immediate:
            param.variable.immediate = True
            param.variable.value = expr.get_step().value
            # We can skip setting the bit in module.vars_set since there is no
            # way to test those bits in the code...


class Error_reporter:
    def __init__(self, location, prefix=""):
        self.location = location
        self.prefix = prefix

    def report(self, msg):
        scanner.syntax_error(f"{self.prefix}: {msg}",
                             self.location.lexpos, self.location.lineno,
                             self.location.filename)

    def positional(self):
        if not self.prefix:
            return Error_reporter(self.location, "Positional params")
        return Error_reporter(self.location,
                              f"{self.prefix}: Positional params")

    def keyword(self, keyword):
        if isinstance(keyword, scanner.Token):
            keyword = keyword.value
        if not self.prefix:
            return Error_reporter(self.location, f"{keyword} params")
        return Error_reporter(self.location, f"{self.prefix}: {keyword} params")

    def param_number(self, n):
        assert self.prefix[-1] == 's'
        return Error_reporter(self.location, f"{self.prefix[:-1]} {n}")

    def return_type(self):
        if not self.prefix:
            return Error_reporter(self.location, "Return")
        return Error_reporter(self.location, f"{self.prefix}: Return")

