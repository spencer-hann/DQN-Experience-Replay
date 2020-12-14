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
    delay = start_delay

    for i in range(delay):
        yield val
    del delay

    while val > end:
        yield val
        val += rate
    del val, rate

    while True:
        yield end

