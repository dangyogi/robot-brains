# parser.py

import ply.yacc as yacc

from scanner import tokens, syntax_error
from symtable import (
    push, pop, current_entity, lookup, Module, Opmode, Subroutine, Function,
    Use, Typedef,
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
    pos_arguments_no_uminus :
    kw_arguments :
    simple_exprs :
    simple_statements :
    '''
    p[0] = ()


def p_first(p):
    '''
    simple_expr_no_uminus : STRING
                          | FLOAT
                          | INTEGER
                          | BOOL
                          | lvalue
    simple_expr : simple_expr_no_uminus
    simple_expr : uminus_expr
    expr : simple_expr
    statement : simple_statement NEWLINE
              | dlt
    lvalue : IDENT
    pos_arguments : pos1_arguments
    pos_arguments_no_uminus : pos1_arguments_no_uminus
    block : statements1
    simple_exprs : simple1_exprs
    simple1_exprs : simple_expr
    '''
    p[0] = p[1]


def p_second(p):
    '''
    simple_expr_no_uminus : '(' expr ')'
    returning_opt : RETURNING idents
    taking_opt : TAKING idents
    '''
    p[0] = p[2]


def p_none(p):
    '''
    file : opmode
         | module
    opmode : IDENT OPMODE push_opmode NEWLINE uses
    module : IDENT MODULE push_module parameters NEWLINE \
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
    fn_decl : FUNCTION IDENT make_fn parameters set_returning_opt NEWLINE \
              typedefs vartypes first_block labeled_blocks pop
    sub_decl : SUBROUTINE IDENT make_sub parameters NEWLINE \
               typedefs vartypes first_block labeled_blocks pop
    first_block :
    labeled_blocks :
    labeled_blocks : labeled_blocks labeled_block
    returning_opt :
    taking_opt :
    '''
    p[0] = None


def p_1tuple(p):
    '''
    conditions : condition
    actions : action
    idents : IDENT
    pos1_arguments : simple_expr
    pos1_arguments_no_uminus : simple_expr_no_uminus
    statements1 :
    '''
    p[0] = (p[1],)


def p_append(p):
    '''
    kw_arguments : kw_arguments kw_argument
    statements1 : statements1 statement
    conditions : conditions condition
    actions : actions action
    kw_argument : KEYWORD pos1_arguments
    simple_statements : simple_statements simple_statement NEWLINE
    '''
    p[0] = p[1] + (p[2],)


def p_append2(p):
    '''
    idents : idents ',' IDENT
    pos1_arguments : pos1_arguments ',' simple_expr
    pos1_arguments_no_uminus : pos1_arguments_no_uminus ',' simple_expr
    simple1_exprs : simple1_exprs ',' simple_expr
    '''
    p[0] = p[1] + (p[3],)


def p_all(p):
    """
    arguments : pos_arguments kw_arguments
    arguments_no_uminus : pos_arguments_no_uminus kw_arguments
    simple_expr_no_uminus : QUOTE lvalue
                          | NOT simple_expr
                          | simple_expr_no_uminus '^' simple_expr
                          | simple_expr_no_uminus '*' simple_expr
                          | simple_expr_no_uminus '/' simple_expr
                          | simple_expr_no_uminus '%' simple_expr
                          | simple_expr_no_uminus '+' simple_expr
                          | simple_expr_no_uminus '-' simple_expr
                          | simple_expr_no_uminus '<' simple_expr
                          | simple_expr_no_uminus LEQ simple_expr
                          | simple_expr_no_uminus LAEQ simple_expr
                          | simple_expr_no_uminus '>' simple_expr
                          | simple_expr_no_uminus GEQ simple_expr
                          | simple_expr_no_uminus GAEQ simple_expr
                          | simple_expr_no_uminus EQ simple_expr
                          | simple_expr_no_uminus AEQ simple_expr
                          | simple_expr_no_uminus NEQ simple_expr
                          | simple_expr_no_uminus NAEQ simple_expr
    uminus_expr : '-' simple_expr  %prec UMINUS
                | uminus_expr '^' simple_expr
                | uminus_expr '*' simple_expr
                | uminus_expr '/' simple_expr
                | uminus_expr '%' simple_expr
                | uminus_expr '+' simple_expr
                | uminus_expr '-' simple_expr
                | uminus_expr '<' simple_expr
                | uminus_expr LEQ simple_expr
                | uminus_expr LAEQ simple_expr
                | uminus_expr '>' simple_expr
                | uminus_expr GEQ simple_expr
                | uminus_expr GAEQ simple_expr
                | uminus_expr EQ simple_expr
                | uminus_expr AEQ simple_expr
                | uminus_expr NEQ simple_expr
                | uminus_expr NAEQ simple_expr
    simple_statement : PASS
                     | lvalue arguments_no_uminus
                     | PREPARE lvalue arguments_no_uminus
                     | REUSE simple_expr
                     | RELEASE lvalue
                     | lvalue '=' expr
                     | idents '=' lvalue arguments_no_uminus
                     | lvalue OPEQ expr
                     | lvalue arguments_no_uminus RETURNING_TO simple_expr
                     | REUSE simple_expr RETURNING_TO simple_expr
                     | GOTO simple_expr pos_arguments
                     | RETURN simple_exprs
                     | RETURN simple_exprs TO simple_expr
    lvalue : lvalue '.' IDENT
    lvalue : lvalue '[' expr ']'
    """
    p[0] = tuple(p)


def p_expr(p):
    'expr : lvalue arguments_no_uminus'
    if any(p[2]):
        p[0] = ('call', p[1], p[2])
    else:
        p[0] = p[1]


def p_pushopmode(p):
    'push_opmode :'
    push(Opmode(p[-2]))


def p_pushmodule(p):
    'push_module :'
    push(Module(p[-2]))


def p_parameter1(p):
    '''
    parameter : IDENT
    '''
    current_entity().pos_parameter(p[1])


def p_parameter2(p):
    '''
    parameter : IDENT '=' simple_expr
    '''
    current_entity().pos_parameter(p[1], p[3])


def p_keyword(p):
    'keyword : KEYWORD'
    current_entity().kw_parameter(p[1])


def p_use1(p):
    '''
    use : USE IDENT arguments NEWLINE
    '''
    p[2].generator = Use(p[2], p[3])


def p_use2(p):
    '''
    use : USE IDENT AS IDENT arguments NEWLINE
    '''
    p[2].generator = Use(p[4], p[5])


def p_typedef(p):
    '''
    typedef : TYPE IDENT IS FUNCTION taking_opt returning_opt NEWLINE
            | TYPE IDENT IS SUBROUTINE taking_opt NEWLINE
    '''
    p[1].generator = Typedef(*p[4:-1])


def p_set_returning_opt(p):
    'set_returning_opt : returning_opt'
    if p[1] is not None:
        current_entity().return_types = p[1]
    p[0] = None


def p_vartype(p):
    '''
    vartype : IDENT IS IDENT NEWLINE
    '''
    p[1].type = p[3]


def p_dim(p):
    '''
    vartype : DIM IDENT '[' expr ']' NEWLINE
    '''
    p[2].dim = p[4]


def p_make_fn(p):
    'make_fn :'
    lookup('global', generator=current_entity())
    p[-1].generator = push(Function(p[-1]))


def p_make_sub(p):
    'make_sub :'
    lookup('global', generator=current_entity())
    p[-1].generator = push(Subroutine(p[-1]))


def p_first_block(p):
    'first_block : block'
    current_entity().add_block(p[1])


def p_labeled_block(p):
    'labeled_block : LABEL IDENT pos_parameters NEWLINE block'
    current_entity().labeled_block(p[2], p[3], p[5])


def p_dlt(p):
    '''
    dlt : DLT_DELIMITER NEWLINE \
          conditions \
          DLT_DELIMITER NEWLINE \
          actions \
          DLT_DELIMITER NEWLINE
    '''
    p[0] = ('dlt', p[2], p[4])


def p_condition(p):
    '''
    condition : expr REST NEWLINE
    '''
    p[0] = (p[1], p[2])


def p_action(p):
    '''
    action : simple_statement REST NEWLINE simple_statements
    '''
    p[0] = p[1], p[4], p[2]


def p_pop(p):
    'pop :'
    pop()
    p[0] = None


def p_error(t):
    syntax_error(t, "p_error: Syntax Error")


parser = yacc.yacc()


if __name__ == "__main__":
    import sys

    parser
