# scanner.py

import sys
import math
import os.path
from fractions import Fraction
import ply.lex as lex


Namespaces = []
Num_errors = 0

Expanded_kws = False


def set_expanded_kws(value):
    global Expanded_kws
    Expanded_kws = value


class Token:
    def __init__(self, lex_token, value=None, type=None, lexpos=None,
                 lineno=None, filename=None):
        if value is None:
            self.value = lex_token.value
        else:
            self.value = value
        self.type = type
        self.lineno = lineno or lex_token.lineno
        if lexpos is None:
            self.lexpos = lex_token.lexpos
        else:
            self.lexpos = lexpos
        if filename is None:
            self.filename = lexer.filename
        else:
            self.filename = filename

    @classmethod
    def dummy(cls, value, type='integer'):
        return cls(None, value=value, type=type, lexpos=0, lineno=1,
                   filename='dummy')

    def __repr__(self):
        return (f"<Token {self.value!r}"
                #f" lineno={self.lineno} lexpos={self.lexpos}"
                ">")

    # Make these usuable as dict keys to preserve the location info for
    # KEYWORDS.  This also does the lower() calls to make the comparisons
    # case-insensitive.
    def __hash__(self):
        if self.value[-1] == ':' and self.value[0] == '?':
            return hash(self.value[1:].lower())
        return hash(self.value.lower())

    def __eq__(self, b):
        if self.value[-1] == ':' and self.value[0] == '?':
            v1 = self.value[1:]
        else:
            v1 = self.value
        if isinstance(b, Token):
            v2 = b.value
        else:
            v2 = b
        if v2[-1] == ':' and v2[0] == '?':
            v2 = v2[1:]
        return v1.lower() == v2.lower()


reserved = frozenset((
    'ABS',
    'AS',
    'AUTONOMOUS',
    'BOOLEAN',
    'CONTINUE',
    'DIM',
    'DONE',
    'FLOAT',
    'FUNCTION',
    'GOTO',
    'INTEGER',
    'IS',
    'LABEL',
    'MODULE',
    'NOT',
    'OPMODE',
    'RETURN',
    'RETURN_LABEL',
    'RETURNING',
    'SET',
    'STRING',
    'SUBROUTINE',
    'TAKING',
    'TELEOP',
    'TYPE',
    'USE',
    'VAR',
))

tokens = (
    'AEQ',
    'BOOLEAN_LIT',
    'DLT_DELIMITER',
    'DLT_MAP',
    'DLT_MASK',
    'EQ',
    'FLOAT_LIT',
    'FROM',      # FROM:  expanded_kw
    'GAEQ',
    'GEQ',
    'GOT',
    'IDENT',
    'INTEGER_DIVIDE',
    'INTEGER_LIT',
    'KEYWORD',
    'LAEQ',
    'LEQ',
    'NAEQ',
    'NATIVE_STRING_LIT',
    'NEQ',
    'NEWLINE',
    'OPEQ',
    'OPT_KEYWORD',
    'RETURNING_TO',
    'STRING_LIT',
    'TO',        # TO:  expanded_kw
    'WITH',      # WITH:  expanded_kw

) + tuple(reserved)


scales = {
    # The annotated number is multiplied by the number in this table to
    # normalize it.

    # Lengths normalize to inches:
    'IN': 1,
    'FT': 12,
    'M': 1/0.0254,
    'CM': 1/2.54,
    'MM': 1/25.4,

    # Time normalizes to seconds:
    'MIN': 60,
    'SEC': 1,
    'MSEC': Fraction(1, 10**3),
    'USEC': Fraction(1, 10**6),

    # Speeds normalize to in/sec:
    'IPS': 1,
    'IPM': Fraction(1, 60),
    'FPS': 12,
    'FPM': Fraction(12, 60),
    'MPS': 1/0.0254,
    'MPM': 1/0.0254/60,
    'MPH': 1/0.0254/60,

    # Accelerations normalize to in/sec^2
    'GRAVITY': 386.08858,

    # Forces normalize to lbs:
    'LB': 1,
    'LBS': 1,
    'OZ': Fraction(1, 16),
    'NEWTON': 0.22480894,
    'NEWTONS': 0.22480894,
    'N': 0.22480894,
    'DYNE': 2.2480894e-6,
    'DYNES': 2.2480894e-6,

    # Angles normalize to degrees:
    'DEG': 1,
    'RAD': 360/(2*math.pi),
    'ROT': 360,

    # Mass normalize to lbs
    'G': 1/453.59237,
    'GRAMS': 1/453.59237,
    'GRAM': 1/453.59237,
    'KG': 2.2046226,
    'KGRAM': 2.2046226,
    'KGRAMS': 2.2046226,

    # Angular speed normalizes to degrees/sec:
    'RPS': 360,
    'RPM': 360//60,

    # Percent normalizes to 0-1
    '%': Fraction(1, 100),
}


t_INITIAL_ignore = ' '

t_ANY_ignore_COMMENT = r'\#.*'
t_ANY_ignore_empty_DLT_MASK = r'\|\ +\|'


literals = "+-*/%^?<>.,()[]{}"


def t_CONTINUATION(t):
    r'\n[ ]*->'
    #print("t_CONTINUATION", repr(t.value), 
    #      "at", t.lineno, t.lexpos)
    t.lexer.lineno += 1
    # No token returned, ie., swallow NEWLINE


def t_NEWLINE(t):
    r'\n+'
    #print("t_NEWLINE", repr(t.value), "at", t.lineno, t.lexpos)
    t.lexer.lineno += len(t.value)
    t.value = Token(t)
    return t


def t_FLOAT_LIT(t):
    r'''(?P<num>[-+]?(\d+\.\d*([eE][-+]?\d+)?
              |\.\d+([eE][-+]?\d+)?
              |\d+[eE][-+]?\d+))(?P<conv>[a-zA-Z%]+(^\d+)?(/[a-zA-Z]+(^\d+)?)?)?
    '''
    conv = t.lexer.lexmatch.group('conv')

    # This is always a float, even if convert returns a Fraction...
    t.value = float(t.lexer.lexmatch.group('num')) * convert(conv, t.lexpos,
                                                             t.lineno)

    return t


def t_INTEGER_LIT(t):
    r'(?P<num>[-+]?\d+)(?P<conv>[a-zA-Z%]+(^\d+)?(/[a-zA-Z]+(^\d+)?)?)?'
    conv = t.lexer.lexmatch.group('conv')

    # This may result in an int, float or Fraction...
    ans = int(t.lexer.lexmatch.group('num')) * convert(conv, t.lexpos, t.lineno)

    if isinstance(ans, float):
        t.type = 'FLOAT_LIT'
        t.value = float(ans)
    elif isinstance(ans, Fraction):
        if ans.denominator == 1:
            t.value = int(ans)
        else:
            t.type = 'FLOAT_LIT'
            t.value = float(ans)
    else:
        t.value = ans
    return t


def t_STRING_LIT(t):
    r'"([^"\n]*|"")*"'
    t.value = t.value[1:-1].replace('""', '"')
    return t


def t_NATIVE_STRING_LIT(t):
    r'`[^`\n]*`'
    t.value = t.value[1:-1]
    return t


def t_KEYWORD(t):
    r'[a-zA-Z_][a-zA-Z_0-9]*:'
    if t.value.lower() == 'returning_to:':
        t.type = 'RETURNING_TO'
        return t
    if Expanded_kws:
        if t.value.lower() == 'from:':
            t.type = 'FROM'
            return t
        if t.value.lower() == 'to:':
            t.type = 'TO'
            return t
        if t.value.lower() == 'with:':
            t.type = 'WITH'
            return t
    t.value = Token(t)
    return t


def t_OPT_KEYWORD(t):
    r'\?[a-zA-Z_][a-zA-Z_0-9]*:'
    t.value = Token(t)
    return t


def t_STRING_IDENT(t):
    r'[a-zA-Z_][a-zA-Z_0-9]*\$'
    t.value = Token(t, type='string')
    t.type = 'IDENT'
    return t


def t_BOOL_IDENT(t):
    r'[a-zA-Z_][a-zA-Z_0-9]*\?'
    uname = t.value.upper()
    if uname == 'TRUE?':
        t.value = True
        t.type = 'BOOLEAN_LIT'
    elif uname == 'FALSE?':
        t.value = False
        t.type = 'BOOLEAN_LIT'
    elif uname == 'GOT?':
        t.type = 'GOT'
    else:
        t.value = Token(t, type='boolean')
        t.type = 'IDENT'
    return t


def t_INT_IDENT(t):
    r'[i-nI-N][a-zA-Z_0-9]*'
    upper = t.value.upper()
    if upper in reserved:
        t.type = upper
    elif upper in scales:
        t.value = scales[upper]
        if isinstance(t.value, int):
            t.type = 'INTEGER_LIT'
        else:
            t.value = float(t.value)
            t.type = 'FLOAT_LIT'
    else:
        t.value = Token(t, type='integer')
        t.type = 'IDENT'
    return t


def t_FLOAT_IDENT(t):
    r'[a-ho-zA-HO-Z_][a-zA-Z_0-9]*'
    upper = t.value.upper()
    if upper in reserved:
        t.type = upper
    elif upper in scales:
        t.value = scales[upper]
        if isinstance(t.value, int):
            t.type = 'INTEGER_LIT'
        else:
            t.value = float(t.value)
            t.type = 'FLOAT_LIT'
    elif upper == 'PI':
        t.value = math.pi
        t.type = 'FLOAT_LIT'
    else:
        t.value = Token(t, type='float')
        t.type = 'IDENT'
    return t


def t_DLT_DELIMITER(t):
    r'={4,}'
    return t


def t_DLT_MASK(t):
    r'\|\ *[-yYnN]+\ *\|'
    t.value = Token(t, t.value[1:-1].rstrip(), lexpos=t.lexpos + 1)
    return t


def t_DLT_MAP(t):
    r'\|\ *[xX][ xX]*\|'
    t.value = Token(t, t.value[1:-1].rstrip(), lexpos=t.lexpos + 1)
    if t.value:
        return t


def t_INTEGER_DIVIDE(t):
    r'//'
    return t


def t_EQ(t):
    r'=='
    return t


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


def t_AEQ(t):
    r'~='
    return t


def t_NEQ(t):
    r'!=|<>'
    return t


def t_NAEQ(t):
    r'!~=|<~>'
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
    upper = terms[0].upper()
    if upper not in scales:
        syntax_error(f"Unknown conversion unit", lexpos, lineno)
    ans = scales[upper]
    if len(terms) == 2:
        try:
            ans **= int(terms[1])
        except ValueError:
            syntax_error(f"Illegal exponent in conversion, integer expected",
                         lexpos + len(terms[0]) + 1,
                         lineno)
    return ans


def find_line_start(lexpos, lexdata=None):
    r'''Return the index of the first character of the line.
    '''
    return (lexdata or lexer.lexdata).rfind('\n', 0, lexpos) + 1


def find_line_end(lexpos, lexdata=None):
    r'''Return the index of the last (non-newline) character of the line.

    Return is < 0 if this is the last line and it doesn't have a newline
    at the end.
    '''
    return (lexdata or lexer.lexdata).find('\n', lexpos) - 1


def find_column(lexpos, lexdata=None):
    ans = (lexpos - find_line_start(lexpos, lexdata)) + 1
    #print(f"find_column({lexpos}) -> {ans}")
    return ans


def find_line(lexpos, lexdata=None):
    if lexdata is None:
        lexdata = lexer.lexdata
    start = find_line_start(lexpos, lexdata)
    end = find_line_end(lexpos, lexdata)
    #print(f"find_line({lexpos}): start {start}, end {end}")
    if end < 0:
        return lexdata[start:]
    else:
        return lexdata[start:end + 1]


def syntax_error(msg, lexpos, lineno, filename = None):
    global Num_errors

    print(f'File "{filename or lexer.filename}", line {lineno}',
          file=sys.stderr)
    if filename is None or filename == lexer.filename:
        #print("syntax_error using lexer.lexdata")
        lexdata = lexer.lexdata
    else:
        #print(f"syntax_error reading {filename} for lexdata")
        with open(filename) as f:
            lexdata = f.read()
    print(" ", find_line(lexpos, lexdata), file=sys.stderr)
    print(" " * find_column(lexpos, lexdata), "^", file=sys.stderr)
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


lexer = lex.lex()


def lex_file(filename=None):
    global Namespaces, Num_errors

    if filename is None:
        lexer.input(sys.stdin.read())
        lexer.filename = "<stdin>"
    else:
        lexer.filename = filename
        with open(filename) as f:
            lexer.input(f.read())

    lexer.lineno = 1

    Namespaces = []
    Num_errors = 0
    return lexer


def check_for_errors():
    if Num_errors:
        print(file=sys.stderr)
        print("Number of Errors:", Num_errors, file=sys.stderr)
        sys.exit(1)


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
