# method_test.py


class foo:
    def __init__(self, name):
        self.name = name


def bogus(self):
    print("bogus", self.name)


before = foo('before')
foo.bar = bogus
after = foo('after')

before.bar()
after.bar()
