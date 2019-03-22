# parser.py

import ply.yacc as yacc

from scanner import tokens, syntax_error, lex_file, Token, check_for_errors
from symtable import (
    pop_namespace, current_namespace, top_namespace, Module, Opmode,
    Subroutine, Function, Use, Typedef, Labeled_block, Block,
    DLT, Conditions, Actions, Variable, Required_parameter, Optional_parameter,
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
    expr : primary
    primary : STRING
            | FLOAT
            | INTEGER
            | BOOL
            | ident_ref
    statement : simple_statement newlines
              | dlt
    pos_arguments : pos1_arguments
    exprs : exprs1
    exprs1 : expr
    '''
    p[0] = p[1]


def p_second(p):
    '''
    primary : '(' expr ')'
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
    idents : IDENT
    '''
    p[0] = (p[1],)


def p_2tuple(p):
    '''
    lvalues : lvalue ',' lvalue
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
    lvalues : lvalues ',' lvalue
    '''
    p[0] = p[1] + (p[3],)


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
    simple_statement : PASS
                     | primary arguments
                     | PREPARE primary arguments
                     | REUSE expr
                     | RELEASE primary
                     | SET lvalue kw_to expr
                     | UNPACK lvalues kw_from expr
                     | primary OPEQ expr
                     | primary arguments RETURNING_TO expr
                     | REUSE expr RETURNING_TO expr
                     | GOTO primary pos_arguments
                     | RETURN exprs
                     | RETURN exprs TO expr
    lvalue : primary '.' IDENT
    lvalue : primary '[' expr ']'
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


def p_primary(p):
    '''
    primary : '{' primary arguments '}'
    '''
    p[0] = ('get', p[2], p[3])


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
    Opmode()


def p_make_module(p):
    'make_module :'
    Module()


def p_parameter1(p):
    '''
    parameter : IDENT
    '''
    Required_parameter(p[1])


def p_parameter2(p):
    '''
    parameter : IDENT '=' expr
    '''
    Optional_parameter(p[1], p[3])


def p_keyword(p):
    'keyword : KEYWORD'
    current_namespace().kw_parameter(p[1])


def p_kw_to(p):
    'kw_to : KEYWORD'
    if p[1].value.lower() != 'to:':
        syntax_error("Expected 'TO:'", p[1].lexpos, p[1].lineno)


def p_kw_from(p):
    'kw_from : KEYWORD'
    if p[1].value.lower() != 'from:':
        syntax_error("Expected 'FROM:'", p[1].lexpos, p[1].lineno)


def p_use1(p):
    '''
    use : USE ident_ref arguments newlines
    '''
    Use(p[2], p[2], p[3])


def p_use2(p):
    '''
    use : USE ident_ref AS IDENT arguments newlines
    '''
    Use(p[2], p[4], p[5])


def p_typedef(p):
    '''
    typedef : TYPE ident_ref IS FUNCTION taking_opt returning_opt newlines
            | TYPE ident_ref IS SUBROUTINE taking_opt newlines
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
    vartype : DIM IDENT '[' expr ']' newlines
    '''
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

