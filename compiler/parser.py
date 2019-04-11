# parser.py

import operator
import os.path
import ply.yacc as yacc

from scanner import tokens, syntax_error, lex_file, Token, check_for_errors
from symtable import (
    pop_namespace, current_namespace, top_namespace, Module, Opmode,
    Subroutine, Function, Use, Typedef, Labeled_block, Block,
    DLT, Conditions, Actions, Variable, Required_parameter, Optional_parameter,
    Statement, Call_statement, Opeq_statement,
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
    ('left', '*', '%'),
    ('right', 'UMINUS'),
    ('right', '^'),
    ('left', '.'),
)


def p_empty_tuple(p):
    '''
    pos_arguments :
    kw_arguments :
    primarys :
    returning_opt :
    taking_opt :
    '''
    p[0] = ()


def p_first(p):
    '''
    const_expr : STRING
               | FLOAT
               | INTEGER
               | BOOL
    expr : primary
    primary : STRING
            | FLOAT
            | INTEGER
            | BOOL
            | ident_ref
    statement : simple_statement newlines
              | dlt
    pos_arguments : pos1_arguments
    opmode_type : AUTONOMOUS
                | TELEOP
    '''
    p[0] = p[1]


def p_second(p):
    '''
    primary : '(' expr ')'
    const_expr : '(' const_expr ')'
    returning_opt : RETURNING idents
    taking_opt : TAKING idents
    '''
    p[0] = p[2]


def p_none(p):
    '''
    newlines_opt :
    newlines_opt : newlines_opt NEWLINE
    newlines : NEWLINE
    newlines : newlines NEWLINE
    opmode : opmode_type OPMODE make_opmode newlines uses
    module : MODULE make_module parameters newlines \
             uses typedefs vartypes decls
    typedefs :
             | typedefs typedef
    vartypes :
             | vartypes vartype
    decls :
          | decls fn_decl
          | decls sub_decl
    parameters : pos_parameters kw_parameters
    pos1_parameters : parameter
    pos1_parameters : pos1_parameters parameter
    pos_parameters :
                   | pos1_parameters
    kw_parameters :
                  | kw_parameters keyword pos1_parameters
    uses :
         | uses use
    fn_decl : FUNCTION fn_name parameters set_returning_opt newlines \
              typedefs vartypes first_block labeled_blocks pop
    sub_decl : SUBROUTINE sub_name parameters newlines \
               typedefs vartypes first_block labeled_blocks pop
    first_block :
    labeled_blocks :
    labeled_blocks : labeled_blocks labeled_block
    labeled_block : LABEL labeled_block_name pos_parameters newlines \
                    first_block
    '''
    p[0] = None


def p_1tuple(p):
    '''
    conditions : condition
    actions : action
    pos1_arguments : primary
    statements1 : statement
    idents : IDENT
    '''
    p[0] = (p[1],)


def p_append(p):
    '''
    kw_arguments : kw_arguments kw_argument
    statements1 : statements1 statement
    conditions : conditions condition
    actions : actions action
    idents : idents IDENT
    pos1_arguments : pos1_arguments primary
    primarys : primarys primary
    '''
    p[0] = p[1] + (p[2],)


def p_all(p):
    """
    arguments : pos_arguments kw_arguments
    kw_argument : KEYWORD pos1_arguments
    expr : NOT expr
         | expr '^' expr
         | expr '*' expr
         | expr '/' expr
         | expr '%' expr
         | expr '+' expr
         | expr '-' expr
         | '-' expr               %prec UMINUS
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
    primary : primary '.' IDENT
            | primary '[' expr ']'
    lvalue : primary '.' IDENT
    lvalue : primary '[' expr ']'
    """
    p[0] = tuple(p[1:])


def p_const_numeric_expr_uminus(p):
    r"""
    const_expr : '-' const_expr               %prec UMINUS
    """
    if not isinstance(p[2], (int, float)):
        syntax_error("Must have numeric operand for unary negate",
                     p[2].lexpos, p[2].lineno)
    p[0] = -p[2]


def p_const_numeric_expr_binary(p):
    r"""
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
    r"""
    const_expr : NOT const_expr
    """
    if not isinstance(p[2], bool):
        syntax_error("Must have bool operand for unary NOT",
                     p[2].lexpos, p[2].lineno)
    p[0] = not p[2]


def p_simple_statement1(p):
    r"""
    simple_statement : PASS
                     | PREPARE primary arguments
                     | REUSE primary
                     | REUSE primary RETURNING_TO primary
                     | RELEASE primary
                     | SET lvalue kw_to primary
                     | GOTO primary pos_arguments
                     | RETURN primarys
                     | RETURN primarys kw_to primary
    """
    p[0] = Statement(*p[1:])


def p_simple_statement2(p):
    r"""
    simple_statement : primary arguments
                     | primary arguments RETURNING_TO primary
    """
    p[0] = Call_statement(*p[1:])


def p_simple_statement3(p):
    r"""
    simple_statement : primary OPEQ primary
    """
    p[0] = Opeq_statement(p[1], p[2], p[3])


def p_file(p):
    r'''
    file : newlines_opt opmode
         | newlines_opt module
    '''
    p[0] = top_namespace()


def p_block(p):
    'block : statements1'
    p[0] = Block(p[1])


def p_primary(p):
    '''
    primary : '{' primary arguments '}'
    '''
    p[0] = ('call_fn', p[2], p[3])


def p_ident_ref(p):
    'ident_ref : IDENT'
    current_namespace().ident_ref(p[1])
    p[0] = p[1]


def p_lvalue(p):
    'lvalue : IDENT'
    current_namespace().assigned_to(p[1])
    p[0] = p[1]


def p_make_opmode(p):
    'make_opmode :'
    Opmode(Modules_seen)


def p_make_module(p):
    'make_module :'
    module = Module()
    name = module.name.value
    assert name not in Modules_seen
    #print("make_module seen", name)
    Modules_seen[name] = module


def p_parameter1(p):
    '''
    parameter : IDENT
    '''
    Required_parameter(p[1])


def p_parameter2(p):
    '''
    parameter : IDENT '=' const_expr
    '''
    Optional_parameter(p[1], p[3])


def p_keyword(p):
    'keyword : KEYWORD'
    current_namespace().kw_parameter(p[1])


def p_kw_to(p):
    'kw_to : KEYWORD'
    if p[1].value.lower() != 'to:':
        syntax_error("Expected 'TO:'", p[1].lexpos, p[1].lineno)
    p[0] = p[1]


def p_use1(p):
    '''
    use : USE ident_ref arguments newlines
    '''
    Use(p[2], p[2], p[3])
    name = p[2].value
    if name not in Modules_seen:
        #print("Use adding", name)
        Modules_needed.append(p[2])


def p_use2(p):
    '''
    use : USE ident_ref AS IDENT arguments newlines
    '''
    Use(p[4], p[2], p[5])
    name = p[2].value
    if name not in Modules_seen:
        #print("Use adding", name)
        Modules_needed.append(p[2])


def p_typedef(p):
    '''
    typedef : TYPE ident_ref IS FUNCTION taking_opt returning_opt newlines
            | TYPE ident_ref IS SUBROUTINE taking_opt newlines
            | TYPE ident_ref IS LABEL taking_opt newlines
    '''
    Typedef(p[2], *p[4:-1])


def p_set_returning_opt(p):
    'set_returning_opt : returning_opt'
    if p[1] is not None:
        current_namespace().set_return_types(p[1])


def p_vartype(p):
    '''
    vartype : IDENT IS IDENT newlines
    '''
    Variable(p[1], type=p[3], no_prior_use=True)


def p_dim(p):
    '''
    vartype : DIM IDENT '[' const_expr ']' newlines
    '''
    if not isinstance(p[4], int):
        syntax_error("Must have integer DIM", p[4].lexpos, p[4].lineno)
    current_entity = current_namespace().lookup(p[2], error_not_found=False)
    if isinstance(current_entity, Reference) or \
       isinstance(current_entity, Variable) and current_entity.assignment_seen:
        syntax_error("Variable referenced before DIM", p[2].lexpos, p[2].lineno)
    if not isinstance(current_entity, Variable):
        current_entity = Variable(p[2])
    current_entity.dim = p[4]


def p_fn_name(p):
    'fn_name : ident_ref'
    Function(p[1])


def p_sub_name(p):
    'sub_name : ident_ref'
    Subroutine(p[1])


def p_first_block(p):
    'first_block : block'
    current_namespace().add_first_block(p[1])


def p_labeled_block_name(p):
    'labeled_block_name : ident_ref'
    Labeled_block(p[1])


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
    condition : expr REST
    '''
    p[0] = p[1], p[2]


def p_action1(p):
    '''
    action : simple_statement NEWLINE
    '''
    p[0] = p[1], Token(p[2], value='')


def p_action2(p):
    '''
    action : simple_statement REST
    '''
    p[0] = p[1], p[2]


def p_dlt_conditions(p):
    'dlt_conditions : conditions'
    p[0] = Conditions(p[1])


def p_dlt_actions(p):
    'dlt_actions : actions'
    p[0] = Actions(p[1])


def p_pop(p):
    'pop :'
    pop_namespace()


def p_error(t):
    print("p_error", t)
    syntax_error("p_error: Syntax Error", t.lexpos, t.lineno)


parser = yacc.yacc()  # (start='opmode')


def parse_opmode(filename, debug=False):
    dirname = os.path.dirname(filename)
    path = [dirname]
    libsdir = os.path.join(os.path.dirname(os.path.abspath(dirname)), "libs")
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

    ans = parse(filename, 'opmode', debug)

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
                     ref_token.lexpos, ref_token.lineno, ref_token.filename,
                     ref_token.namespace)
    return full_name


def _find_module(name, path):
    #print("_find_module", name, path)
    for p in path:
        full_name = os.path.join(p, name + '.module')
        #print("_find_module", name, "checking", full_name)
        if os.path.isfile(full_name):
            return full_name
    #print("_find_module:", name, "not found")
    return None


if __name__ == "__main__":
    import sys

    assert len(sys.argv) == 2
    filename = sys.argv[1]
    top_entity = parse_opmode(filename, # debug=True,
                             )
    print("top_entity is", top_entity)
    print()
    print("dump:")
    print()
    top_entity.dump(sys.stdout)

    print()
    print("code:")
    print()
    import C_gen
    C_gen.gen_program(top_entity)

