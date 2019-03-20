# scanner.py

import math
from fractions import Fraction
import ply.lex as lex

from symtable import lookup


reserved = frozenset((
    'AS',
    'DIM',
    'FUNCTION',
    'GOTO',
    'IS',
    'LABEL',
    'MODULE',
    'NOT',
    'OPMODE',
    'PASS',
    'PREPARE',
    'RELEASE',
    'RETURN',
    'RETURNING',
    'REUSE',
    'SUBROUTINE',
    'TAKING',
    'TO',
    'TYPE',
    'USE',

))

tokens = (
    'AEQ',
    'BOOL',
    'DLT_DELIMITER',
    'EQ',
    'FLOAT',
    'GAEQ',
    'GEQ',
    'IDENT',
    'INTEGER',
    'KEYWORD',
    'LAEQ',
    'LEQ',
    'NAEQ',
    'NEQ',
    'NEWLINE',
    'OPEQ',
    'QUOTE',
    'REST',
    'RETURNING_TO',
    'STRING',

) + tuple(reserved)


scales = {
    # Lengths normalize to inches:
    'in': 1,
    'ft': 12,
    'm': 1/0.0254,
    'cm': 1/2.54,
    'mm': 1/25.4,

    # Time normalizes to seconds:
    'min': 60,
    'sec': 1,
    'msec': Fraction(1, 10**3),
    'usec': Fraction(1, 10**6),

    # Speeds normalize to in/sec:
    'ips': 1,
    'ipm': Fraction(1, 60),
    'fps': 12,
    'fpm': Fraction(12, 60),
    'mps': 1/0.0254,
    'mpm': 1/0.0254/60,

    # Forces normalize to lbs:
    'lb': 1,
    'lbs': 1,
    'oz': Fraction(1, 16),
    'g': 1/453.59237,
    'kg': 2.2046226,

    # Angles normalize to degrees:
    'deg': 1,
    'rad': 360/(2*math.pi),
    'rot': 360,

    # Angular speed normalizes to degrees/sec:
    'rps': 360,
    'rpm': 6, # 360/60

    # Percent normalizes to 0-1
    '%': Fraction(1, 100),
}


Symtables = []


t_INITIAL_ignore = ' '

t_ANY_ignore_COMMENT = r'\#.*'


literals = "+-*/%^=<>.()[]"

last_rest = None


def t_CONTINUATION(t):
    r'\n[ ]*->'
    #print("t_CONTINUATION", repr(t.value), last_rest,
    #      "at", t.lexer.lineno, t.lexpos)
    t.lexer.lineno += 1
    # No token returned, ie., swallow NEWLINE


def t_NEWLINE(t):
    r'\n+'
    global last_rest
    #print("t_NEWLINE", repr(t.value), "at", t.lexer.lineno, t.lexpos)
    if last_rest is not None:
        #print("t_NEWLINE returning last_rest", last_rest)
        t = last_rest
        last_rest = None
    t.lexer.lineno += len(t.value)
    return t


def t_FLOAT(t):
    r'''(?P<num>[-+]?(\d+\.\d*([eE][-+]?\d+)?
              |\.\d+([eE][-+]?\d+)?
              |\d+[eE][-+]?\d+))(?P<conv>[a-zA-Z%]+(^\d+)?(/[a-zA-Z]+(^\d+)?)*)?
    '''
    conv = t.lexer.lexmatch.group('conv')

    # This is always a float, even if convert returns a Fraction...
    t.value = float(t.lexer.lexmatch.group('num')) * convert(t, conv)

    return t


def t_INTEGER(t):
    r'(?P<num>[-+]?\d+)(?P<conv>[a-zA-Z%]+(^\d+)?(/[a-zA-Z]+(^\d+)?)*)?'
    conv = t.lexer.lexmatch.group('conv')

    # This may result in an int, float or Fraction...
    ans = int(t.lexer.lexmatch.group('num')) * convert(t, conv)

    if isinstance(ans, float):
        t.type = 'FLOAT'
        t.value = float(ans)
    elif isinstance(ans, Fraction):
        if ans.denominator == 1:
            t.value = int(ans)
        else:
            t.type = 'FLOAT'
            t.value = float(ans)
    else:
        t.value = ans
    return t


def t_STRING(t):
    r'"([^"]*|"")*"'
    t.value = t.value[1:-1].replace('""', '"')
    return t


def t_KEYWORD(t):
    r'[a-zA-Z_][a-zA-Z_0-9]*:'
    t.value = t.value[:-1]
    if t.value.lower() == 'returning_to':
        t.type = 'RETURNING_TO'
    return t


def t_STRING_IDENT(t):
    r'[a-zA-Z_][a-zA-Z_0-9]*\$'
    t.value = lookup(t.value[:-1], type='str')
    t.type = 'IDENT'
    return t


def t_BOOL_IDENT(t):
    r'[a-zA-Z_][a-zA-Z_0-9]*\?'
    name = t.value[:-1]
    if name.lower() == 'true':
        t.value = True
        t.type = 'BOOL'
    elif name.lower() == 'false':
        t.value = False
        t.type = 'BOOL'
    else:
        t.value = lookup(name, type='bool')
        t.type = 'IDENT'
    return t


def t_INT_IDENT(t):
    r'[i-nI-N][a-zA-Z_0-9]*'
    upper = t.value.upper()
    if upper in reserved:
        t.type = upper
    else:
        t.value = lookup(t.value, type='int')
        t.type = 'IDENT'
    return t


def t_FLOAT_IDENT(t):
    r'[a-ho-zA-HO-Z_][a-zA-Z_0-9]*'
    upper = t.value.upper()
    if upper in reserved:
        t.type = upper
    else:
        t.value = lookup(t.value, type='float')
        t.type = 'IDENT'
    return t


def t_DLT_DELIMITER(t):
    r'={4,}'
    return t


def t_REST(t):
    r'\|[^#\n]*'
    global last_rest
    t.value = t.value[1:]
    if t.value.strip():
        if last_rest is None:
            last_rest = t
            #print("t_REST: setting last_rest to", last_rest)
        else:
            syntax_error(t, "not allowed on continuation line", 1)


def t_GEQ(t):
    r'>='
    return t


def t_GAEQ(t):
    r'>~='
    return t


def t_LEQ(t):
    r'<='
    return t


def t_LAEQ(t):
    r'<~='
    return t


def t_EQ(t):
    r'=='
    return t


def t_AEQ(t):
    r'~='
    return t


def t_NEQ(t):
    r'!=|<>'
    return t


def t_NAEQ(t):
    r'!~=|<~>'
    return t


def t_QUOTE(t):
    r"'"
    return t


def t_OPEQ(t):
    r"[-+*/%^]="
    return t


def t_error(t):
    print(f"WARNING {t.lexer.filename}[{t.lexer.lineno}]: "
          f"Illegal character '{t.value[0]}' -- ignored")
    t.lexer.skip(1)


def convert(t, conversion):
    if conversion is None: return 1.0
    segments = conversion.split('/')
    #print("convert", conversion, segments)
    ans = convert_segment(t, segments[0])
    for seq in segments[1:]:
        ans /= convert_segment(seg)
    return ans


def convert_segment(t, seg):
    terms = seg.split('^')
    #print("convert_segment", seg, terms)
    if len(terms) > 2:
        syntax_error(t, "multiple exponents in unit conversion not allowed")
    if terms[0] not in scales:
        syntax_error(t, f"unknown unit, {terms[0]}")
    ans = scales[terms[0]]
    if len(terms) == 2:
        try:
            ans **= int(terms[1])
        except ValueError:
            syntax_error(t, f"illegal exponent, {terms[1]}, integer expected")
    return ans


def find_line_start(token):
    return token.lexer.lexdata.rfind('\n', 0, token.lexpos) + 1


def find_line_end(token):
    return token.lexer.lexdata.find('\n', 0, token.lexpos) - 1


def find_column(token):
    return (token.lexpos - find_line_start(token)) + 1


def find_line(token):
    return token.lexer.lexdata[find_line_start(token):find_line_end(token)]


def syntax_error(token, msg, column_offset = 0):
    raise SyntaxError(msg, (token.lexer.filename,
                            token.lexer.lineno,
                            find_column(token) + column_offset,
                            find_line(token)))


lexer = lex.lex()
lexer.filename = "<stdin>"


if __name__ == "__main__":
    import sys

    print("got", len(sys.argv), "args", "argv[0] is", sys.argv[0])

    if len(sys.argv) > 1:
        lexer.filename = sys.argv[1]
        lexer.input(open(sys.argv[1]).read())
    else:
        lexer.input(sys.stdin.read())

    while True:
        tok = lexer.token()
        if not tok:
            break
        print(tok)
