from sys import stdin


class BinaryTree:
    def __init__(self, key, parent=None, left_child=None, right_child=None):
        self._key = key
        self._parent = parent
        self._left = left_child
        self._right = right_child

    @classmethod
    def build_from_list(cls, lst, *, root_index=0):
        created = [cls(key) for key, *_ in lst]
        for parent, (_, left_i, right_i) in zip(created, lst):
            if left_i != -1:
                left = created[left_i]
                parent._left = left
                left._parent = parent
            if right_i != -1:
                right = created[right_i]
                parent._right = right
                right._parent = parent
        return created[root_index]

    def in_order_iter(self):
        if self._left is not None:
            yield from self._left.in_order_iter()
        yield self
        if self._right is not None:
            yield from self._right.in_order_iter()

    def pre_order_iter(self):
        yield self
        if self._left is not None:
            yield from self._left.pre_order_iter()
        if self._right is not None:
            yield from self._right.pre_order_iter()

    def post_order_iter(self):
        if self._left is not None:
            yield from self._left.post_order_iter()
        if self._right is not None:
            yield from self._right.post_order_iter()
        yield self


def main():
    _ = stdin.readline()
    lst = [list(map(int, line.split())) for line in stdin]
    tree = BinaryTree.build_from_list(lst)
    print(*[node._key for node in tree.in_order_iter()])
    print(*[node._key for node in tree.pre_order_iter()])
    print(*[node._key for node in tree.post_order_iter()])


def get_all_orders(nodes):
    tree = BinaryTree.build_from_list(nodes)
    return [
        [node._key for node in tree.in_order_iter()],
        [node._key for node in tree.pre_order_iter()],
        [node._key for node in tree.post_order_iter()]
    ]


def test():
    assert get_all_orders([
        [4, 1, 2],
        [2, 3, 4],
        [5, -1, -1],
        [1, -1, -1],
        [3, -1, -1]
    ]) == [
        [1, 2, 3, 4, 5],
        [4, 2, 1, 3, 5],
        [1, 3, 2, 5, 4]
    ]
    assert get_all_orders([
        [0, 7, 2],
        [10, -1, -1],
        [20, -1, 6],
        [30, 8, 9],
        [40, 3, -1],
        [50, -1, -1],
        [60, 1, -1],
        [70, 5, 4],
        [80, -1, -1],
        [90, -1, -1]
    ]) == [
        [50, 70, 80, 30, 90, 40, 0, 20, 10, 60],
        [0, 70, 50, 40, 30, 80, 90, 20, 60, 10],
        [50, 80, 90, 30, 40, 70, 10, 60, 20, 0]
    ]
    assert get_all_orders([[98, -1, -1]]) == [
        [98], [98], [98]
    ]


def time_test(number=1):
    import gc
    import random
    import sys
    import timeit
    # node_number = 10 ** 5
    node_number = 10 ** 3

    def prepare():
        nodes = []
        for i in range(node_number):
            key = random.randint(0, 10 ** 9)
            if i < node_number - 1:
                left = i + 1
            else:
                left = - 1
            right = -1
            nodes.append((key, left, right))
        return nodes
    # sys.setrecursionlimit(2*node_number)
    # print(*get_all_orders(prepare()))
    print(
        timeit.timeit(
            stmt="gc.enable(); get_all_orders(prepare())",
            setup="sys.setrecursionlimit(2 * node_number)",
            globals={**globals(), **locals()}, number=number) / number)


if __name__ == "__main__":
    main()
    # test()
    # time_test()
