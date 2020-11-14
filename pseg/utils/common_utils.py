def del_weakref(ref):
    o = ref()
    if o is not None:
        o.__del__()
