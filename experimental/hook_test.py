# hook_test.py

# hook mechanism




class Hook_base:
    hooks = {}    # {(cls, hook_name): fn}

    @classmethod
    def add_hook(cls, hook_name, fn):
        assert (cls, hook_name) not in Hook_base.hooks, \
               f"More than one {hook_name} hook on {cls.__name__}"
        Hook_base.hooks[cls, hook_name] = fn

    def run_hooks(self, hook_name, *args, **kws):
        for c in self.__class__.__mro__:
            fn = Hook_base.hooks.get((c, hook_name))
            if fn is not None:
                return fn(self, *args, **kws)


class base(Hook_base):
    def test(self):
        print(self.__class__.__name__, "test", self.run_hooks("test"))


class derived(base):
    pass


b = base()
d = derived()

b.test()
d.test()

def a_fn(self):
    return "a"


def b_fn(self):
    return "b"


base.add_hook("test", a_fn)
derived.add_hook("test", b_fn)

b.test()
d.test()
