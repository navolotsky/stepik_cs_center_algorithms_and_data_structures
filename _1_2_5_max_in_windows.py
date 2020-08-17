import sys


class MaxQueue:
    def __init__(self, iterable=None, *, max_size=None):
        self._max_size = max_size
        self._input_stack = []
        self._input_stack_maxs = []
        self._output_stack = []
        self._output_stack_maxs = []
        if iterable is not None:
            for el in iterable:
                self.push(el)

    def cur_size(self):
        return len(self._input_stack) + len(self._output_stack)

    def push(self, el):
        if self._max_size is not None and self.cur_size() == self._max_size:
            self.pop()
        self._input_stack.append(el)
        if not self._input_stack_maxs or el > self._input_stack_maxs[-1]:
            self._input_stack_maxs.append(el)
        else:
            self._input_stack_maxs.append(self._input_stack_maxs[-1])

    def pop(self):
        if self._output_stack:
            self._output_stack_maxs.pop()
            return self._output_stack.pop()
        if not self._input_stack:
            return None
        self._input_stack_maxs.clear()
        while self._input_stack:
            el = self._input_stack.pop()
            self._output_stack.append(el)
            if not self._output_stack_maxs or el > self._output_stack_maxs[-1]:
                self._output_stack_maxs.append(el)
            else:
                self._output_stack_maxs.append(self._output_stack_maxs[-1])
        self._output_stack_maxs.pop()
        return self._output_stack.pop()

    def max(self):
        if self._output_stack_maxs and self._input_stack_maxs:
            return max(self._output_stack_maxs[-1], self._input_stack_maxs[-1])
        if self._output_stack_maxs:
            return self._output_stack_maxs[-1]
        if self._input_stack_maxs:
            return self._input_stack_maxs[-1]
        return None


def max_in_windows(array, window_size):
    result = []
    queue = MaxQueue(array[:window_size - 1], max_size=window_size)
    for el in array[window_size - 1:]:
        queue.push(el)
        result.append(queue.max())
    return result


def main():
    _ = sys.stdin.readline()
    array = list(map(int, sys.stdin.readline().split()))
    window = int(sys.stdin.readline())
    print(*max_in_windows(array, window))


def test():
    assert max_in_windows([2, 7, 3, 1, 5, 2, 6, 2], 4) == [7, 7, 5, 6, 6]
    assert max_in_windows([2, 1, 5], 1) == [2, 1, 5]
    assert max_in_windows([2, 3, 9], 3) == [9]
    assert max_in_windows([0], 1) == [0]


def time_test(number=1):
    import timeit
    import random
    import string
    import gc
    n = 10**5
    print(timeit.timeit(
        stmt="max_in_windows(array, window)",
        setup="gc.enable(); array = [random.randint(0, n) for _ in range(n)]; window = random.randint(1, n)",
        globals={**globals(), **locals()}, number=number)/number)


if __name__ == "__main__":
    # main()
    test()
    # time_test()
