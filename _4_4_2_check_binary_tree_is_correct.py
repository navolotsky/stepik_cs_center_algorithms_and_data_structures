from sys import stdin


def is_sorted(lst, *, reversed_=False):
    if not lst:
        return True
    prev = lst[0]
    if reversed_:
        def cmp(x, y): return x >= y
    else:
        def cmp(x, y): return x <= y
    for cur in lst[1:]:
        if not cmp(prev, cur):
            return False
        prev = cur
    return True


class BinaryTree:
    def __init__(self, key=None, parent=None, left_child=None, right_child=None):
        self._key = key
        self._parent = parent
        self._left = left_child
        self._right = right_child

    @classmethod
    def build_from_list(cls, lst, *, root_index=0):
        if not lst:
            return BinaryTree()
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
        if self._key is None:
            return
        yield self
        if self._right is not None:
            yield from self._right.in_order_iter()

    def pre_order_iter(self):
        if self._key is None:
            return
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
        if self._key is None:
            return
        yield self

    def in_order_iter_by_iterative(self):
        cur = self
        visited = set()
        while True:
            if cur._left is not None and cur._left not in visited:
                cur = cur._left
            else:
                if cur not in visited:
                    yield cur
                    visited.add(cur)
                if cur._right is not None and cur._right not in visited:
                    cur = cur._right
                elif cur._parent is not None:
                    visited.discard(cur._left)
                    visited.discard(cur._right)
                    cur = cur._parent
                else:
                    return

    def is_correct(self):
        keys = [node._key for node in self.in_order_iter_by_iterative()]
        return is_sorted(keys)


def main():
    _ = stdin.readline()
    lst = [list(map(int, line.split())) for line in stdin]
    tree = BinaryTree.build_from_list(lst)
    if tree.is_correct():
        print("CORRECT")
    else:
        print("INCORRECT")


def test():
    assert BinaryTree.build_from_list([
        [2, 1, 2],
        [1, -1, -1],
        [3, -1, -1]
    ]).is_correct() == True
    assert BinaryTree.build_from_list([
        [1, 1, 2],
        [2, -1, -1],
        [3, -1, -1]
    ]).is_correct() == False
    assert BinaryTree().is_correct() == True
    assert BinaryTree.build_from_list(
        [[98, -1, -1]]).is_correct() == True
    assert BinaryTree.build_from_list([
        [1, -1, 1],
        [2, -1, 2],
        [3, -1, 3],
        [4, -1, 4],
        [5, -1, -1]
    ]).is_correct() == True
    assert BinaryTree.build_from_list([
        [4, 1, 2],
        [2, 3, 4],
        [6, 5, 6],
        [1, -1, -1],
        [3, -1, -1],
        [5, -1, -1],
        [7, -1, -1]
    ]).is_correct() == True
    assert BinaryTree.build_from_list([
        [4, 1, -1],
        [2, 2, 3],
        [1, -1, -1],
        [5, -1, -1]
    ]).is_correct() == False


def time_test(number=1):
    import gc
    import random
    import sys
    import timeit
    # node_number = 10 ** 5
    node_number = 10 ** 5

    def prepare():
        nodes = []
        for i in range(node_number):
            key = random.randint(-2 ** 31 + 1, 2 ** 31 - 2)
            if i < node_number - 1:
                left = i + 1
            else:
                left = - 1
            right = -1
            nodes.append((key, left, right))
        return nodes
    print(
        timeit.timeit(
            stmt="gc.enable(); BinaryTree.build_from_list(prepare()).is_correct()",
            globals={**globals(), **locals()}, number=number) / number)


if __name__ == "__main__":
    # main()
    test()
    # time_test()
