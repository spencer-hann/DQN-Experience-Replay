from math import abs


def Repeater(a):
    while True:
        yield a


def FallbackRepeater(a):
    if hasattr(a, '__iter__'):
        for _a in a:
            yield _a
    else:
        _a = a

    while True:
        yield _a


def LinearScheduler(val, end, rate=None, start_delay=0):
    if rate is None:
        val, end, rate = 1, val, end

    if abs(rate) > abs(val - end):  # interpret rate as n steps
        rate =  abs(val - end) / rate

    for _ in range(start_delay):
        yield val

    state = (val > end)
    while (val > end) == state:
        yield val
        val += rate

    while True:
        yield end

