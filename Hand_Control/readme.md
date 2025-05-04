In utils.py, add ".abc."

def is_sequence(obj):
    """Test if "obj" is a sequence."""
    return isinstance(obj, collections.abc.Iterable)


def is_three_sequence(obj):
    """Test if "obj" is of a sequence type and three long."""
    return isinstance(obj, collections.abc.Iterable) and len(obj) == 3