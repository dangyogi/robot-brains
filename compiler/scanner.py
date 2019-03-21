# scanner.py

import sys
import math
import os.path
from fractions import Fraction
import ply.lex as lex


Namespaces = []
Num_errors = 0


class Token:
    def __init__(self, value, type=None, lexpos=None, lineno=None):
        self.value = value
        self.type = type
        self.lineno = lineno or lexer.lineno
        if lexpos is None:
            self.lexpos = lexer.lexpos
        else:
            self.lexpos = lexpos

    def __repr__(self):
        return f"<Token {self.value!r} lineno={self.lineno} " \
               f"lexpos={self.lexpos}>"


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


t_INITIAL_ignore = ' '

t_ANY_ignore_COMMENT = r'\#.*'


literals = "+-*/%^=<>.()[]{}"

last_rest = None


def t_CONTINUATION(t):
    r'\n[ ]*->'
    #print("t_CONTINUATION", repr(t.value), last_rest,
    #      "at", t.lineno, t.lexpos)
    t.lexer.lineno += 1
    # No token returned, ie., swallow NEWLINE


def t_NEWLINE(t):
    r'\n+'
    global last_rest
    #print("t_NEWLINE", repr(t.value), "at", t.lineno, t.lexpos)
    t.lexer.lineno += len(t.value)
    if last_rest is not None:
        #print("t_NEWLINE returning last_rest", last_rest)
        t.value = last_rest
        t.type = 'REST'
        last_rest = None
    else:
        t.value = Token(t.value, lexpos=t.lexpos, lineno=t.lineno)
    return t


def t_FLOAT(t):
    r'''(?P<num>[-+]?(\d+\.\d*([eE][-+]?\d+)?
              |\.\d+([eE][-+]?\d+)?
              |\d+[eE][-+]?\d+))(?P<conv>[a-zA-Z%]+(^\d+)?(/[a-zA-Z]+(^\d+)?)*)?
    '''
    conv = t.lexer.lexmatch.group('conv')

    # This is always a float, even if convert returns a Fraction...
    t.value = float(t.lexer.lexmatch.group('num')) * convert(conv, t.lexpos,
                                                             t.lineno)

    return t


def t_INTEGER(t):
    r'(?P<num>[-+]?\d+)(?P<conv>[a-zA-Z%]+(^\d+)?(/[a-zA-Z]+(^\d+)?)*)?'
    conv = t.lexer.lexmatch.group('conv')

    # This may result in an int, float or Fraction...
    ans = int(t.lexer.lexmatch.group('num')) * convert(conv, t.lexpos, t.lineno)

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
    if t.value.lower() == 'returning_to:':
        t.type = 'RETURNING_TO'
    else:
        t.value = Token(t.value)
    return t


def t_STRING_IDENT(t):
    r'[a-zA-Z_][a-zA-Z_0-9]*\$'
    t.value = Token(t.value, 'str')
    t.type = 'IDENT'
    return t


def t_BOOL_IDENT(t):
    r'[a-zA-Z_][a-zA-Z_0-9]*\?'
    name = t.value
    if name.lower() == 'true?':
        t.value = True
        t.type = 'BOOL'
    elif name.lower() == 'false?':
        t.value = False
        t.type = 'BOOL'
    else:
        t.value = Token(name, 'bool')
        t.type = 'IDENT'
    return t


def t_INT_IDENT(t):
    r'[i-nI-N][a-zA-Z_0-9]*'
    upper = t.value.upper()
    if upper in reserved:
        t.type = upper
    else:
        t.value = Token(t.value, 'int')
        t.type = 'IDENT'
    return t


def t_FLOAT_IDENT(t):
    r'[a-ho-zA-HO-Z_][a-zA-Z_0-9]*'
    upper = t.value.upper()
    if upper in reserved:
        t.type = upper
    else:
        t.value = Token(t.value, 'float')
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
            last_rest = Token(t.value, lexpos=t.lexpos + 1, lineno=t.lineno)
            #print("t_REST: got", t.lexpos,
            #      "setting last_rest to", last_rest)
        else:
            syntax_error("Not allowed on continuation line",
                         t.lexpos
                           + 1 + (len(t.value) - len(t.value.lstrip())),
                         t.lineno)


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
    print(f"WARNING {t.lexer.filename}[{t.lineno}]: "
          f"Illegal character '{t.value[0]}' -- ignored")
    t.lexer.skip(1)


def convert(conversion, lexpos, lineno):
    if conversion is None: return 1
    segments = conversion.split('/')
    #print("convert", conversion, segments)
    ans = convert_segment(segments[0], lexpos, lineno)
    lexpos += len(segments[0]) + 1
    for seq in segments[1:]:
        ans /= convert_segment(seg, lexpos, lineno)
        lexpos += len(seq) + 1
    return ans


def convert_segment(seg, lexpos, lineno):
    terms = seg.split('^')
    #print("convert_segment", seg, terms)
    if len(terms) > 2:
        syntax_error("Multiple exponents in unit conversion not allowed",
                     lexpos + len(terms[0]) + len(terms[1]) + 1,
                     lineno)
    if terms[0] not in scales:
        syntax_error(f"Unknown conversion unit", lexpos, lineno)
    ans = scales[terms[0]]
    if len(terms) == 2:
        try:
            ans **= int(terms[1])
        except ValueError:
            syntax_error(f"Illegal exponent in conversion, integer expected",
                         lexpos + len(terms[0]) + 1,
                         lineno)
    return ans


def find_line_start(lexpos):
    r'''Return the index of the first character of the line.
    '''
    return lexer.lexdata.rfind('\n', 0, lexpos) + 1


def find_line_end(lexpos):
    r'''Return the index of the last (non-newline) character of the line.

    Return is < 0 if this is the last line and it doesn't have a newline
    at the end.
    '''
    return lexer.lexdata.find('\n', lexpos) - 1


def find_column(lexpos):
    ans = (lexpos - find_line_start(lexpos)) + 1
    #print(f"find_column({lexpos}) -> {ans}")
    return ans


def find_line(lexpos):
    start = find_line_start(lexpos)
    end = find_line_end(lexpos)
    #print(f"find_line({lexpos}): start {start}, end {end}")
    if end < 0:
        return lexer.lexdata[start:]
    else:
        return lexer.lexdata[start:end + 1]


def syntax_error(msg, lexpos, lineno, filename = None):
    global Num_errors

    print(f'  File "{filename or lexer.filename}", '
            f'line {lineno}',
          end='', file=sys.stderr)
    if Namespaces:
        print(f', in {Namespaces[-1]}', file=sys.stderr)
    else:
        print(file=sys.stderr)
    print("   ", find_line(lexpos), file=sys.stderr)
    print(" " * (2 + find_column(lexpos)), "^", file=sys.stderr)
    print("SyntaxError:", msg)

    Num_errors += 1

    raise SyntaxError

    #raise SyntaxError(msg, (filename or lexer.filename,
    #                        lineno,
    #                        find_column(lexpos),
    #                        find_line(lexpos)))


def filename():
    return lexer.filename


def file_basename():
    return os.path.basename(filename()).rsplit('.', 1)[0]


def lex_file(filename=None):
    global Namespaces, Num_errors

    if filename is None:
        lexer.input(sys.stdin.read())
    else:
        lexer.filename = filename
        with open(sys.argv[1]) as f:
            lexer.input(f.read())

    Namespaces = []
    Num_errors = 0


def check_for_errors():
    if Num_errors:
        print(file=sys.stderr)
        print("Number of Errors:", Num_errors, file=sys.stderr)
        sys.exit(1)


lexer = lex.lex()
lexer.filename = "<stdin>"


if __name__ == "__main__":
    print("got", len(sys.argv), "args", "argv[0] is", sys.argv[0])

    if len(sys.argv) > 1:
        lex_file(sys.argv[1])
    else:
        lex_file()

    while True:
        tok = lexer.token()
        if not tok:
            break
        print(tok)
