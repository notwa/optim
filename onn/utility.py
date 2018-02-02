import sys


def lament(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def lower_priority():
    """Set the priority of the process to below-normal."""
    # via https://stackoverflow.com/a/1023269
    if sys.platform == 'win32':
        try:
            import win32api
            import win32process
            import win32con
            pid = win32api.GetCurrentProcessId()
            handle = win32api.OpenProcess(
                win32con.PROCESS_ALL_ACCESS, True, pid)
            win32process.SetPriorityClass(
                handle, win32process.BELOW_NORMAL_PRIORITY_CLASS)
        except ImportError:
            lament("you do not have pywin32 installed.")
            lament("the process priority could not be lowered.")
            lament("consider: python -m pip install pypiwin32")
            lament("consider: conda install pywin32")
    else:
        import os
        os.nice(1)


def onehot(y):
    unique = np.unique(y)
    Y = np.zeros((y.shape[0], len(unique)), dtype=np.int8)
    offsets = np.arange(len(y)) * len(unique)
    Y.flat[offsets + y.flat] = 1
    return Y


# more

_log_was_update = False


def log(left, right, update=False):
    s = "\x1B[1m  {:>20}:\x1B[0m   {}".format(left, right)
    global _log_was_update
    if update and _log_was_update:
        lament('\x1B[F' + s)
    else:
        lament(s)
    _log_was_update = update


class Dummy:
    pass
