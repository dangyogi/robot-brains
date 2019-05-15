# parser.py

import operator
import os.path
import ply.yacc as yacc

from robot_brains.scanner import (
    tokens, syntax_error, lex_file, Token, check_for_errors, set_expanded_kws,
)

from robot_brains.symtable import (
    last_parameter_obj, current_namespace, top_namespace, Module, Opmode,
    Subroutine, Function, Native_statement, Native_expr,
    Use, Typedef, Label, Return_label, DLT, Conditions, DLT_MAP, Actions,
    Variable, Required_parameter, Optional_parameter,
    Continue, Set, Goto, Return, Call_statement, Opeq_statement, Done_statement,
    Literal, Reference, Dot, Subscript, Got_keyword, Got_param, Call_fn,
    Unary_expr, Binary_expr, Builtin_type, Typename_type, Label_type,
)


Modules_needed = []
Modules_seen = {}


start = 'file'

precedence = (   # lowest to highest
    ('nonassoc', 'NOT'),
    ('nonassoc', 'EQ', 'NEQ'),
    ('nonassoc', '<', 'LEQ', 'LAEQ', '>', 'GEQ', 'GAEQ', 'AEQ', 'NAEQ'),
    ('left', '+', '-'),
    ('right', '/'),
    ('left', '*', '%', 'INTEGER_DIVIDE'),
    ('right', 'UMINUS', 'ABS'),
    ('right', '^'),
    ('left', '.'),
)


def p_empty_tuple(p):
    '''
    pos_arguments :
    kw_arguments :
    parameter_types_list :
    steps :
    statements :
    action_statements :
    dotted_prefix :
    '''
    p[0] = ()


def p_first(p):
    '''
    const_expr : STRING_LIT
               | FLOAT_LIT
               | INTEGER_LIT
               | BOOLEAN_LIT
    expr : primary
    primary : simple_primary
    native_element : primary
                   | NATIVE_STRING_LIT
    type : simple_type
    statement : simple_statement newlines
              | dlt
    actions : action
    parameter_types_list : parameter_types_list1
    opmode_type : AUTONOMOUS
                | TELEOP
    '''
    p[0] = p[1]


def p_second(p):
    '''
    simple_primary : '(' expr ')'
    simple_type : '(' type ')'
    const_expr : '(' const_expr ')'
    returning_opt : RETURNING parameter_types
    taking_opt : TAKING parameter_types
    from_opt : FROM primary
    label_decl : FUNCTION fn_name parameters set_returning_opt newlines
               | SUBROUTINE sub_name parameters newlines
               | LABEL label_name parameters newlines
    '''
    p[0] = p[2]


def p_none(p):
    '''
    newlines_opt :
    newlines_opt : newlines_opt NEWLINE
    newlines : NEWLINE
    newlines : newlines NEWLINE
    opmode : opmode_type OPMODE make_opmode newlines uses typedefs
    typedefs :
             | typedefs typedef
    vartypes :
             | vartypes vartype
    parameters : pos_parameters kw_parameters
    pos_parameters : required_parameters
    pos_parameters : required_parameters '?' optional_parameters
    required_parameters :
    required_parameters : required_parameters required_parameter
    optional_parameters : optional_parameter
    optional_parameters : optional_parameters optional_parameter
    kw_parameters :
                  | kw_parameters keyword pos_parameters
    returning_opt :
    uses :
         | uses use
    taking_opt :
    from_opt :
    '''
    p[0] = None


def p_1tuple(p):
    '''
    conditions : condition
    parameter_types_list1 : simple_type
    kw_parameter_types : kw_parameter_type
    dimensions : dimension
    subscripts : expr
    native_elements : NATIVE_STRING_LIT
    '''
    p[0] = (p[1],)


def p_append(p):
    '''
    kw_arguments : kw_arguments kw_argument
    statements : statements statement
    conditions : conditions condition
    pos_arguments : pos_arguments primary
    kw_parameter_types : kw_parameter_types kw_parameter_type
    parameter_types_list1 : parameter_types_list1 simple_type
    action_statements : action_statements simple_statement newlines
    action_statements : action_statements continue newlines
    native_elements : native_elements native_element
    dotted_prefix : dotted_prefix IDENT '.'
    '''
    p[0] = p[1] + (p[2],)


def p_dimensions(p):
    '''
    dimensions : dimensions ',' dimension
    subscripts : subscripts ',' expr
    '''
    p[0] = p[1] + (p[3],)


def p_actions(p):
    'actions : actions action'
    p[0] = p[1] + p[2]


def p_step1(p):
    '''
    step : label_decl typedefs vartypes statements
    '''
    p[0] = (p[1],) + p[4]


def p_paste(p):
    '''
    steps : steps step
    '''
    p[0] = p[1] + p[2]


def p_all(p):
    """
    arguments : pos_arguments kw_arguments
    kw_argument : KEYWORD pos_arguments
    parameter_types : pos_parameter_types kw_parameter_types
    """
    p[0] = tuple(p[1:])


def p_kw_parameter_type(p):
    '''
    kw_parameter_type : KEYWORD pos_parameter_types
    kw_parameter_type : OPT_KEYWORD pos_parameter_types
    '''
    p[0] = (p[1], p[2])


def p_module(p):
    '''
    module : MODULE make_module parameters newlines \
             uses typedefs vartypes steps
    '''
    current_namespace().add_steps(p[8])


def p_label_type1(p):
    """
    type : SUBROUTINE taking_opt
         | LABEL taking_opt
    """
    p[0] = Label_type(p[1], p[2])


def p_label_type2(p):
    """
    type : FUNCTION taking_opt returning_opt
    """
    p[0] = Label_type(p[1], p[2], p[3])


def p_builtin_type(p):
    """
    simple_type : INTEGER
                | FLOAT
                | BOOLEAN
                | STRING
                | MODULE
                | TYPE
    """
    p[0] = Builtin_type(p[1].lower())


def p_typename(p):
    "simple_type : dotted_prefix IDENT"
    p[0] = Typename_type(p[1] + (p[2],))


def p_dimension(p):
    '''
    dimension : const_expr
    '''
    if not isinstance(p[1], int):
        syntax_error("Must have integer DIM", p[1].lexpos, p[1].lineno)
    p[0] = p[1]


def p_primary_literal(p):
    """
    primary : STRING_LIT
            | FLOAT_LIT
            | INTEGER_LIT
            | BOOLEAN_LIT
    """
    p[0] = Literal(p.lexpos(1), p.lineno(1), p[1])


def p_return_label1(p):
    """
    simple_primary : RETURN_LABEL
    """
    p[0] = Return_label(p.lexpos(1), p.lineno(1))


def p_return_label2(p):
    """
    simple_primary : simple_primary '.' RETURN_LABEL
    """
    p[0] = Return_label(p.lexpos(3), p.lineno(3), p[1])


def p_primary_ident(p):
    """
    simple_primary : IDENT
    lvalue : IDENT
    """
    p[0] = Reference(p[1])


def p_primary_dot(p):
    """
    simple_primary : simple_primary '.' IDENT
    lvalue : simple_primary '.' IDENT
    """
    p[0] = Dot(p[1], p[3])


def p_primary_subscript(p):
    """
    simple_primary : simple_primary '[' subscripts ']'
    lvalue : simple_primary '[' subscripts ']'
    """
    p[0] = Subscript(p[1], p[3])


def p_primary_got_keyword1(p):
    "primary : GOT KEYWORD"
    p[0] = Got_keyword(p.lexpos(1), p.lineno(1), p[2])


def p_primary_got_keyword2(p):
    "primary : GOT IDENT '.' KEYWORD"
    p[0] = Got_keyword(p.lexpos(1), p.lineno(1), p[4], p[2])


def p_primary_got_keyword3(p):
    "primary : GOT MODULE '.' KEYWORD"
    p[0] = Got_keyword(p.lexpos(1), p.lineno(1), p[4], module=True)


def p_primary_got_param1(p):
    "primary : GOT IDENT"
    p[0] = Got_param(p.lexpos(1), p.lineno(1), p[2])


def p_primary_got_param2(p):
    "primary : GOT IDENT '.' IDENT"
    p[0] = Got_param(p.lexpos(1), p.lineno(1), p[4], p[2])


def p_primary_got_param3(p):
    "primary : GOT MODULE '.' IDENT"
    p[0] = Got_param(p.lexpos(1), p.lineno(1), p[4], module=True)


def p_unary_expr(p):
    """
    expr : ABS expr
         | NOT expr
         | '-' expr               %prec UMINUS
    """
    p[0] = Unary_expr(p.lexpos(1), p.lineno(1), p[1], p[2])


def p_binary_expr(p):
    """
    expr : expr '^' expr
         | expr '*' expr
         | expr '/' expr
         | expr INTEGER_DIVIDE expr
         | expr '%' expr
         | expr '+' expr
         | expr '-' expr
         | expr '<' expr
         | expr LEQ expr
         | expr LAEQ expr
         | expr '>' expr
         | expr GEQ expr
         | expr GAEQ expr
         | expr EQ expr
         | expr AEQ expr
         | expr NEQ expr
         | expr NAEQ expr
    """
    p[0] = Binary_expr(p[1], p[2], p[3])


def p_native_expr(p):
    """
    expr : native_elements
    """
    p[0] = Native_expr(p.lexpos(1), p.lineno(1), p[1])


def p_pos_parameter_types1(p):
    '''
    parameter_types : pos_parameter_types1
    pos_parameter_types1 : parameter_types_list1
    pos_parameter_types : parameter_types_list
    '''
    p[0] = p[1], ()


def p_pos_parameter_types2(p):
    '''
    pos_parameter_types1 : parameter_types_list '?' parameter_types_list1
    pos_parameter_types : parameter_types_list '?' parameter_types_list1
    '''
    p[0] = p[1], p[3]


def p_const_numeric_expr_uminus(p):
    """
    const_expr : '-' const_expr               %prec UMINUS
    """
    if not isinstance(p[2], (int, float)):
        syntax_error("Must have numeric operand for unary negate",
                     p[2].lexpos, p[2].lineno)
    p[0] = -p[2]


def p_const_numeric_expr_binary(p):
    """
    const_expr : const_expr '^' const_expr
               | const_expr '*' const_expr
               | const_expr '/' const_expr
               | const_expr '%' const_expr
               | const_expr '+' const_expr
               | const_expr '-' const_expr
    """
    if not isinstance(p[1], (int, float)):
        syntax_error(f"Must have numeric operand for {p[2]}",
                     p[1].lexpos, p[1].lineno)
    if not isinstance(p[3], (int, float)):
        syntax_error(f"Must have numeric operand for {p[2]}",
                     p[3].lexpos, p[3].lineno)
    p[0] = ops[p[2]](p[1], p[3])

ops = {
    '^': operator.pow,
    '*': operator.mul,
    '/': lambda a, b: a // b if a % b == 0 else a / b,
    '%': operator.mod,
    '+': operator.add, 
    '-': operator.sub, 
}


def p_const_bool_expr_uminus(p):
    """
    const_expr : NOT const_expr
    """
    if not isinstance(p[2], bool):
        syntax_error("Must have boolean operand for unary NOT",
                     p[2].lexpos, p[2].lineno)
    p[0] = not p[2]


def p_simple_statement1(p):
    """
    continue : CONTINUE
    """
    p[0] = Continue(p.lexpos(1), p.lineno(1))


def p_simple_statement2(p):
    """
    simple_statement : SET extended_kws lvalue TO normal_kws primary
    """
    p[0] = Set(p.lexpos(1), p.lineno(1), p[3], p[6])


def p_simple_statement3(p):
    """
    simple_statement : GOTO primary arguments
    """
    p[0] = Goto(p.lexpos(1), p.lineno(1), p[2], p[3])


def p_simple_statement4(p):
    """
    simple_statement : extended_kws RETURN arguments from_opt normal_kws
    """
    p[0] = Return(p.lexpos(2), p.lineno(2), p[3], p[4], None)


def p_simple_statement5(p):
    """
    simple_statement : extended_kws RETURN arguments from_opt TO primary \
                       normal_kws
    """
    p[0] = Return(p.lexpos(2), p.lineno(2), p[3], p[4], p[6])


def p_extended_kws(p):
    'extended_kws : '
    set_expanded_kws(True)


def p_normal_kws(p):
    'normal_kws : '
    set_expanded_kws(False)


def p_simple_statement6(p):
    """
    simple_statement : primary arguments
    """
    p[0] = Call_statement(p.lexpos(1), p.lineno(1), p[1], p[2], None)


def p_simple_statement7(p):
    """
    simple_statement : primary arguments RETURNING_TO primary
    """
    p[0] = Call_statement(p.lexpos(1), p.lineno(1), p[1], p[2], p[4])


def p_simple_statement8(p):
    """
    simple_statement : lvalue OPEQ primary
    """
    p[0] = Opeq_statement(p.lexpos(1), p.lineno(1), p[1], p[2], p[3])


def p_simple_statement9(p):
    """
    simple_statement : extended_kws DONE normal_kws
    """
    p[0] = Done_statement(p.lexpos(2), p.lineno(2), None)


def p_simple_statement10(p):
    """
    simple_statement : extended_kws DONE WITH IDENT normal_kws
    """
    p[0] = Done_statement(p.lexpos(2), p.lineno(2), p[4])


def p_simple_statement11(p):
    """
    simple_statement : native_elements
    """
    p[0] = Native_statement(p.lexpos(1), p.lineno(1), p[1])


def p_file(p):
    '''
    file : newlines_opt opmode
         | newlines_opt module
    '''
    p[0] = top_namespace()


def p_primary(p):
    '''
    simple_primary : '{' primary arguments '}'
    '''
    p[0] = Call_fn(p[2], p[3][0], p[3][1])


def p_make_opmode(p):
    'make_opmode :'
    Opmode(Modules_seen)


def p_make_module(p):
    'make_module :'
    module = Module()
    name = module.name
    assert name not in Modules_seen
    #print("make_module seen", name)
    Modules_seen[name] = module


def p_required_parameter(p):
    '''
    required_parameter : IDENT
    '''
    Required_parameter(p[1])


def p_optional_parameter(p):
    '''
    optional_parameter : IDENT
    '''
    Optional_parameter(p[1])


def p_keyword(p):
    'keyword : KEYWORD'
    last_parameter_obj().kw_parameter(p[1])


def p_opt_keyword(p):
    "keyword : OPT_KEYWORD"
    last_parameter_obj().kw_parameter(p[1], optional=True)


def p_use1(p):
    '''
    use : USE IDENT arguments newlines
    '''
    Use(p[2], p[2], p[3][0], p[3][1])
    name = p[2].value
    if name not in Modules_seen:
        #print("Use adding", name)
        Modules_needed.append(p[2])


def p_use2(p):
    '''
    use : USE IDENT AS IDENT arguments newlines
    '''
    Use(p[4], p[2], p[5][0], p[5][1])
    name = p[2].value
    if name not in Modules_seen:
        #print("Use adding", name)
        Modules_needed.append(p[2])


def p_typedef(p):
    '''
    typedef : TYPE IDENT IS type newlines
    '''
    Typedef(p[2], p[4])


def p_set_returning_opt(p):
    'set_returning_opt : returning_opt'
    if p[1] is not None:
        last_parameter_obj().set_return_types(p[1])


def p_vartype(p):
    '''
    vartype : VAR IDENT IS type newlines
    '''
    var = current_namespace().make_variable(p[2])
    if var.explicit_typedef:
        syntax_error("Duplicate variable type declaration",
                     p[2].lexpos, p[2].lineno)
    var.type = p[4]
    var.explicit_typedef = True


def p_dim(p):
    '''
    vartype : DIM IDENT '[' dimensions ']' newlines
    '''
    var = current_namespace().make_variable(p[2])
    if var.dimensions:
        syntax_error("Duplicate variable DIM declaration",
                     p[2].lexpos, p[2].lineno)
    var.dimensions = p[4]


def p_fn_name(p):
    'fn_name : IDENT'
    p[0] = Function(p[1])


def p_sub_name(p):
    'sub_name : IDENT'
    p[0] = Subroutine(p[1])


def p_label_name(p):
    'label_name : IDENT'
    p[0] = Label(p[1])


def p_dlt(p):
    '''
    dlt : DLT_DELIMITER NEWLINE \
          dlt_conditions \
          DLT_DELIMITER NEWLINE \
          dlt_actions \
          DLT_DELIMITER newlines
    '''
    p[0] = DLT(p[3], p[6])


def p_condition(p):
    '''
    condition : DLT_MASK expr newlines
    '''
    p[0] = p[1], p[2]


def p_action1(p):
    '''
    action : DLT_MAP action_statements
    '''
    p[0] = (DLT_MAP(p[1]),) + p[2]


def p_action2(p):
    '''
    action : DLT_MAP newlines action_statements
    '''
    p[0] = (DLT_MAP(p[1]),) + p[3]


def p_dlt_conditions(p):
    'dlt_conditions : conditions'
    p[0] = Conditions(p.lineno(1), p[1])


def p_dlt_actions(p):
    'dlt_actions : actions'
    p[0] = Actions(p[1])


def p_error(t):
    print("p_error", t)
    syntax_error("p_error: Syntax Error", t.lexpos, t.lineno)


parser = yacc.yacc()  # (start='opmode')


def parse_opmode(filename, debug=False):
    global Rootdir

    full_name = os.path.abspath(filename)
    dirname = os.path.dirname(full_name)
    path = [dirname]
    Rootdir = os.path.dirname(dirname)
    libsdir = os.path.join(Rootdir, "libs")
    if os.path.isdir(libsdir):
        for entry in os.scandir(libsdir):
            if entry.is_dir():
                path.append(entry.path)
    #print("path", path)

    def parse(filename, file_type='module', debug=False):
        #print("parse", filename, file_type)
        lexer = lex_file(filename)
        ans = parser.parse(lexer=lexer, tracking=True, debug=debug)
        check_for_errors()
        assert ans is not None
        return ans

    ans = parse(full_name, 'opmode', debug)

    # always parse 'builtins' module
    full_name = _find_module('builtins', path)
    if full_name is None:
        print("'builtins' module not found", file=sys.stderr)
        raise SyntaxError
    parse(full_name, debug=debug)

    # parse all other explicitly mentioned modules
    module_parsed = True
    while module_parsed:
        module_parsed = False
        for m in Modules_needed:
            if m.value not in Modules_seen:
                parse(find_module(m, path), debug=debug)
                module_parsed = True

    # return opmode module
    return ans


def find_module(ref_token, path):
    full_name = _find_module(ref_token.value, path)
    if full_name is None:
        syntax_error(f"module {ref_token.value} not found",
                     ref_token.lexpos, ref_token.lineno, ref_token.filename)
    return full_name


def _find_module(name, path):
    #print("_find_module", name, path)
    for p in path:
        full_name = os.path.join(p, name + '.module')
        #print("_find_module", name, "checking", full_name)
        if os.path.isfile(full_name):
            #print("_find_module", name, "found", full_name)
            return full_name
    #print("_find_module:", name, "not found")
    return None


def compile():
    import sys

    print("recursion limit was", sys.getrecursionlimit(), "setting to 100")
    sys.setrecursionlimit(100)

    assert len(sys.argv) == 2
    filename = sys.argv[1]
    opmode = parse_opmode(filename, # debug=True,
                             )
    print("opmode is", opmode)
    if False:
        print()
        print("dump:")
        print()
        opmode.dump(sys.stdout)

    print()
    print("generate:")
    print()
    from robot_brains.code_generator import init_code_generator, generate
    from robot_brains import C_gen
    init_code_generator(C_gen, Rootdir)

    try:
        generate(opmode)
    except SyntaxError:
        #raise
        sys.exit(1)


if __name__ == "__main__":
    compile()
