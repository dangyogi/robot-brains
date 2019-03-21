# parser.py

import ply.yacc as yacc

from scanner import tokens, syntax_error, lex_file, Token, check_for_errors
from symtable import (
    pop_namespace, current_namespace, top_namespace, Module, Opmode,
    Subroutine, Function, Use, Typedef, Labeled_block, Block,
    DLT, Conditions, Actions, 
)


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
    exprs :
    '''
    p[0] = ()


def p_first(p):
    '''
    expr : STRING
         | FLOAT
         | INTEGER
         | BOOL
         | lvalue
    statement : simple_statement newlines
              | dlt
    lvalue : IDENT
    pos_arguments : pos1_arguments
    exprs : exprs1
    exprs1 : expr
    '''
    p[0] = p[1]


def p_second(p):
    '''
    expr : '(' expr ')'
    returning_opt : RETURNING idents
    taking_opt : TAKING idents
    '''
    p[0] = p[2]


def p_none(p):
    '''
    newlines : NEWLINE
    newlines : newlines NEWLINE
    opmode : OPMODE make_opmode newlines uses
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
    pos1_parameters : pos1_parameters ',' parameter
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
    returning_opt :
    taking_opt :
    '''
    p[0] = None


def p_1tuple(p):
    '''
    conditions : condition
    actions : action
    pos1_arguments : expr
    statements1 : statement
    '''
    p[0] = (p[1],)


def p_2tuple(p):
    '''
    idents : IDENT ',' IDENT
    '''
    p[0] = (p[1], p[3])


def p_append(p):
    '''
    kw_arguments : kw_arguments kw_argument
    statements1 : statements1 statement
    conditions : conditions condition
    actions : actions action
    '''
    p[0] = p[1] + (p[2],)


def p_append2(p):
    '''
    idents : idents ',' IDENT
    pos1_arguments : pos1_arguments ',' expr
    exprs1 : exprs1 ',' expr
    '''
    p[0] = p[1] + (p[3],)


def p_all(p):
    """
    arguments : pos_arguments kw_arguments
    kw_argument : KEYWORD pos1_arguments
    expr : QUOTE lvalue
         | NOT expr
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
    simple_statement : PASS
                     | lvalue arguments
                     | PREPARE lvalue arguments
                     | REUSE expr
                     | RELEASE lvalue
                     | lvalue '=' expr
                     | idents '=' lvalue arguments
                     | lvalue OPEQ expr
                     | lvalue arguments RETURNING_TO expr
                     | REUSE expr RETURNING_TO expr
                     | GOTO lvalue pos_arguments
                     | RETURN exprs
                     | RETURN exprs TO expr
    lvalue : lvalue '.' IDENT
    lvalue : lvalue '[' expr ']'
    """
    p[0] = tuple(p)


def p_file(p):
    r'''
    file : opmode
         | module
    '''
    p[0] = top_namespace()


def p_block(p):
    'block : statements1'
    p[0] = Block(p[1])


def p_expr(p):
    '''
    expr : '{' lvalue arguments '}'
    '''
    p[0] = ('get', p[2], p[3])


def p_make_opmode(p):
    'make_opmode :'
    Opmode()


def p_make_module(p):
    'make_module :'
    Module()


def p_parameter1(p):
    '''
    parameter : IDENT
    '''
    current_namespace().pos_parameter(p[1])


def p_parameter2(p):
    '''
    parameter : IDENT '=' expr
    '''
    current_namespace().pos_parameter(p[1], p[3])


def p_keyword(p):
    'keyword : KEYWORD'
    current_namespace().kw_parameter(p[1])


def p_use1(p):
    '''
    use : USE IDENT arguments newlines
    '''
    Use(p[2], p[2], p[3])


def p_use2(p):
    '''
    use : USE IDENT AS IDENT arguments newlines
    '''
    Use(p[2], p[4], p[5])


def p_typedef(p):
    '''
    typedef : TYPE IDENT IS FUNCTION taking_opt returning_opt newlines
            | TYPE IDENT IS SUBROUTINE taking_opt newlines
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
    Variable(p[1], 'local', current_namespace().lookup(p[2]))


def p_dim(p):
    '''
    vartype : DIM IDENT '[' expr ']' newlines
    '''
    Variable(p[2], 'local', 'array of ' + p[2].type).dim = p[4]


def p_fn_name(p):
    'fn_name : IDENT'
    Function(p[1])


def p_sub_name(p):
    'sub_name : IDENT'
    Subroutine(p[1])


def p_first_block(p):
    'first_block : block'
    current_namespace().add_first_block(p[1])


def p_labeled_block_name(p):
    'labeled_block_name : IDENT'
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
    p[0] = p[1], Token('', p[2].lexpos, p[2].lineno)


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


parser = yacc.yacc()


def parse(filename, debug=False):
    lex_file(filename)
    ans = parser.parse(debug=debug)
    check_for_errors()
    assert ans is not None
    return ans


if __name__ == "__main__":
    import sys

    assert len(sys.argv) == 2
    top_entity = parse(sys.argv[1], # debug=True,
                      )
    print("top_entity", top_entity)
    top_entity.dump(sys.stdout)

