import sys


class MaxStack:
    def __init__(self):
        self._values = []
        self._maxes = []

    def push(self, el):
        self._values.append(el)
        if not self._maxes or el > self._maxes[-1]:
            self._maxes.append(el)
        else:
            self._maxes.append(self._maxes[-1])

    def pop(self):
        if self._values:
            self._maxes.pop()
            return self._values.pop()
        return None

    def max(self):
        if self._values:
            return self._maxes[-1]
        return None


def main():
    max_stack = MaxStack()
    _ = sys.stdin.readline()
    for line in sys.stdin.read().splitlines():
        if line.startswith("push"):
            max_stack.push(int(line.split()[1]))
        elif line == "pop":
            max_stack.pop()
        elif line == "max":
            print(max_stack.max())


if __name__ == "__main__":
    main()
