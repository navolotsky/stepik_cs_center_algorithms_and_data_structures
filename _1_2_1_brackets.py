import sys


class Stack:
    def __init__(self):
        self._container = []

    def push(self, value):
        self._container.append(value)

    def pop(self):
        return self._container.pop()

    def empty(self):
        return not bool(self._container)


def is_balanced(string):
    stack = Stack()
    brackets = {
        '}': '{',
        ']': '[',
        ')': '('
    }
    for i, ch in enumerate(string, 1):
        if ch in brackets.values():
            stack.push((ch, i))
        elif ch in brackets.keys():
            if stack.empty():
                return i
            top, _ = stack.pop()
            if top != brackets[ch]:
                return i
    if stack.empty():
        return 'Success'
    else:
        return stack.pop()[1]


def main():
    string = sys.stdin.readline().rstrip()
    print(is_balanced(string))


def test():
    assert is_balanced("([](){([])})") == "Success"
    assert is_balanced("()[]}") == 5
    assert is_balanced("{{[()]]") == 7
    assert is_balanced("{{[()]]") == 7
    assert is_balanced("[]") == "Success"
    assert is_balanced("][") == 1
    assert is_balanced("][") == 1
    assert is_balanced("{[]}()") == "Success"
    assert is_balanced("{") == 1
    assert is_balanced("{[}") == 3
    assert is_balanced("{[}") == 3
    assert is_balanced("foo(bar);") == "Success"
    assert is_balanced("foo(bar[i);") == 10
    assert is_balanced("foo(bar[i;") == 8
    #
    assert is_balanced("([](){([])})") == "Success"
    assert is_balanced("()[]}") == 5
    assert is_balanced("{{[()]]") == 7
    assert is_balanced("{{{[][][]") == 3
    assert is_balanced("{*{{}") == 3
    assert is_balanced("[[*") == 2
    assert is_balanced("{*}") == "Success"
    assert is_balanced("{{") == 2
    assert is_balanced("{}") == "Success"
    assert is_balanced("") == "Success"
    assert is_balanced("}") == 1
    assert is_balanced("*{}") == "Success"
    assert is_balanced("{{{**[][][]") == 3
    assert is_balanced("[(])") == 3


def time_test(number=1000):
    import timeit
    import random
    import string
    import gc
    print(timeit.timeit(stmt="is_balanced(string_)",
                        setup="string_=''.join(random.choices(string.printable, k=10**5))", globals={**globals(), **locals()}, number=1)/number)


if __name__ == "__main__":
    main()
    # test()
    # time_test()
