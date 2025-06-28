def try_import(*modules):
    for mod in modules:
        try:
            return __import__(mod, fromlist=["*"])
        except ImportError:
            continue
    return None
