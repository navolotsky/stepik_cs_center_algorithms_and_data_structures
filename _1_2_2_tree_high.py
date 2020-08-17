import sys


class Tree:
    def __init__(self, parent_list):
        self._container = []
        self._root = None
        if not parent_list:
            return
        for i, parent in enumerate(parent_list):
            self._container.append((parent, []))
            if parent == -1:
                self._root = i
        for i, parent in enumerate(parent_list):
            if parent != -1:
                self._container[parent][1].append(i)

    def _high_recurs(self, vertex):
        return 1 + max(
            (self._high_recurs(v) for v in self._container[vertex][1]),
            default=0
        )

    def high_recurs(self):
        if not self._container:
            return 0
        return self._high_recurs(self._root)

    def high(self):
        if not self._container:
            return 0
        counter = 0
        level = [self._container[self._root][1]]
        while level:
            temp = []
            for level_part in level:
                for vertex in level_part:
                    temp.append(self._container[vertex][1])
            level = temp
            counter += 1
        return counter


def main():
    _ = sys.stdin.readline()
    parent_list = list(map(int, sys.stdin.readline().split()))
    tree = Tree(parent_list)
    print(tree.high())


def test():
    assert Tree([9, 7, 5, 5, 2, 9, 9, 9, 2, -1]).high() == 4
    assert Tree([-1]).high() == 1
    assert Tree([]).high() == 0
    assert Tree([4, -1, 4, 1, 1]).high() == 3
    assert Tree([-1, 0, 4, 0, 3]).high() == 4


def time_test(number=1, size=10**5):
    import timeit
    import random
    import string
    import gc

    print(timeit.timeit(
        stmt="Tree(parent_list).high()",
        setup="parent_list = [random.randrange(size) for _ in range(size)]; parent_list[random.randrange(size)] = -1",
        globals={**globals(), **locals()}, number=number)/number)


if __name__ == "__main__":
    main()
    # test()
    # time_test()
