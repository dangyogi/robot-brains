# str_repr_test.py

class foo:
    def __repr__(self):
        return "foo.repr"

    def __str__(self):
        return "foo.str"


f = foo()

print(f)
print("str", str(f))
print("repr", repr(f))
print("{}", f"{f}")
print("{!r}", f"{f!r}")
print("{!s}", f"{f!s}")
