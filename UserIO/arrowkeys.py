import sys, tty, termios, time


class _Getch:
    def __call__(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch


getch = _Getch()


key_map = {'A':'up', 'B':'down', 'C':'right', 'D':'left'}
key_map = {ord(ch):val for ch, val in key_map.items()}


def get_arrow_key(out_map=None):
    while True:
        ch = getch()
        if ch == '0' or not ch:
            return
        ch = ord(ch)
        if ch not in key_map:
            continue
        key = key_map[ch]
        if out_map:
            return out_map[key]
        return key

