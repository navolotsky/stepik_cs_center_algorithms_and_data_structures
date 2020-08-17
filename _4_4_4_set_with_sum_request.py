from sys import stdin


class Node:
    def __init__(self, number, parent=None, left_child=None, right_child=None):
        self.number = number
        self.parent = parent
        self.left = left_child
        self.right = right_child
        self.high = 0
        self.size = 1
        self.sum = number

    @property
    def key(self):
        return self.number

    def __hash__(self):
        return id(self)


class AVLTree:
    def __init__(self, numbers=None):
        self._root = None
        if numbers is not None:
            for num in numbers:
                self.add(num)

    @classmethod
    def _from_root(cls, root):
        tree = cls()
        tree._root = root
        return tree

    @staticmethod
    def _update_attrs(*nodes):
        for node in nodes:
            if node is None:
                continue
            # update high
            left_high = node.left.high if node.left is not None else -1
            right_high = node.right.high if node.right is not None else -1
            node.high = max(left_high, right_high) + 1
            # update size
            left_size = node.left.size if node.left is not None else 0
            right_size = node.right.size if node.right is not None else 0
            node.size = left_size + right_size + 1
            # update sum
            left_sum = node.left.sum if node.left is not None else 0
            right_sum = node.right.sum if node.right is not None else 0
            node.sum = left_sum + right_sum + node.number

    @classmethod
    def _left_rotation(cls, node):
        # small left rotation
        if (
            (node.right is None and node.left.left is not None) or
            (node.right is not None and node.left.left is not None and
                node.left.left.high + 1 - node.right.high == 2)):
            node_old_left = node.left

            node_new_left = node.left.right
            node.left = node_new_left
            if node_new_left is not None:
                node_new_left.parent = node

            node_old_left.right = node

            node_old_left.parent = node.parent
            if node.parent is not None:
                if node.parent.right is node:
                    node.parent.right = node_old_left
                elif node.parent.left is node:
                    node.parent.left = node_old_left
                else:
                    raise RuntimeError

            node.parent = node_old_left

            cls._update_attrs(node, node_old_left)
            return node_old_left

        # large left rotation
        else:
            node_old_left = node.left

            node_new_left = node.left.right.right
            node_new_parent = node.left.right

            node.left = node_new_left
            if node_new_left is not None:
                node_new_left.parent = node

            node_new_parent.parent = node.parent
            if node.parent is not None:
                if node.parent.right is node:
                    node.parent.right = node_new_parent
                elif node.parent.left is node:
                    node.parent.left = node_new_parent
            node.parent = node_new_parent

            node_new_parent.right = node

            node_old_left_new_right = node_old_left.right.left
            node_old_left.right = node_old_left_new_right
            if node_old_left_new_right is not None:
                node_old_left_new_right.parent = node_old_left

            node_new_parent.left = node_old_left
            node_old_left.parent = node_new_parent

            cls._update_attrs(node_new_parent.left,
                              node_new_parent.right, node_new_parent)
            return node_new_parent

    @classmethod
    def _right_rotation(cls, node):
        # small right rotation
        if (
            (node.left is None and node.right.right is not None) or
            (node.left is not None and node.right.right is not None and
                node.right.right.high + 1 - node.left.high == 2)):
            node_old_right = node.right

            node_new_right = node.right.left
            node.right = node_new_right
            if node_new_right is not None:
                node_new_right.parent = node

            node_old_right.left = node

            node_old_right.parent = node.parent
            if node.parent is not None:
                if node.parent.left is node:
                    node.parent.left = node_old_right
                elif node.parent.right is node:
                    node.parent.right = node_old_right
                else:
                    raise RuntimeError

            node.parent = node_old_right

            cls._update_attrs(node, node_old_right)
            return node_old_right

        # large right rotation
        else:
            node_old_right = node.right

            node_new_right = node.right.left.left
            node_new_parent = node.right.left

            node.right = node_new_right
            if node_new_right is not None:
                node_new_right.parent = node

            node_new_parent.parent = node.parent
            if node.parent is not None:
                if node.parent.left is node:
                    node.parent.left = node_new_parent
                elif node.parent.right is node:
                    node.parent.right = node_new_parent
            node.parent = node_new_parent

            node_new_parent.left = node

            node_old_right_new_left = node_old_right.left.right
            node_old_right.left = node_old_right_new_left
            if node_old_right_new_left is not None:
                node_old_right_new_left.parent = node_old_right

            node_new_parent.right = node_old_right
            node_old_right.parent = node_new_parent

            cls._update_attrs(node_new_parent.left,
                              node_new_parent.right, node_new_parent)
            return node_new_parent

    @classmethod
    def _rebalance(cls, node):
        cls._update_attrs(node)
        left_high = node.left.high if node.left is not None else -1
        right_high = node.right.high if node.right is not None else -1
        maybe_new_root = None
        if right_high - left_high == 2:
            maybe_new_root = cls._right_rotation(node)
        elif left_high - right_high == 2:
            maybe_new_root = cls._left_rotation(node)
        return node if maybe_new_root is None else maybe_new_root

    def _fix(self, node):
        while node is not None:
            maybe_new_root = self._rebalance(node)
            if self._root is node:
                self._root = maybe_new_root
            node = node.parent

    def _insert_after(self, node, new):
        if new.key < node.key:
            node_old_child = node.left
            node.left = new
            new.parent = node
        else:
            node_old_child = node.right
            node.right = new
            new.parent = node
        if node_old_child is not None:
            if node_old_child.key < new.key:
                new.left = node_old_child
            else:
                new.right = node_old_child
            node_old_child.parent = new
        self._fix(new)

    def _add(self, new_node):
        if self._root is None:
            self._root = new_node
            return True
        cur_node = self._root
        while True:
            if cur_node.key == new_node.key:
                return False
            if cur_node.left is not None and cur_node.key > new_node.key:
                cur_node = cur_node.left
            elif cur_node.right is not None and cur_node.key < new_node.key:
                cur_node = cur_node.right
            else:
                self._insert_after(cur_node, new_node)
                return True

    def add(self, number):
        return self._add(Node(number))

    def _find(self, key):
        cur = self._root
        if cur is None:
            return None
        while True:
            if cur.key == key:
                return cur
            if cur.left is not None and cur.key > key:
                cur = cur.left
            elif cur.right is not None and cur.key < key:
                cur = cur.right
            else:
                return None

    def find(self, number):
        return True if self._find(key=number) is not None else False

    @staticmethod
    def _min(node):
        prev, cur = node, node.left
        while cur is not None:
            prev, cur = cur, cur.left
        return prev

    @staticmethod
    def _max(node):
        prev, cur = node, node.right
        while cur is not None:
            prev, cur = cur, cur.right
        return prev

    @classmethod
    def _prev(cls, node):
        # go right and down in left subtree
        if node.left is not None:
            return cls._max(node.left)
        # go right and up while it's possible
        prev, cur = node, node.parent
        while cur is not None and cur.left is prev:
            prev, cur = cur, cur.parent
        return cur if cur is not node else None

    @classmethod
    def _next(cls, node):
        # go left and down in right subtree
        if node.right is not None:
            return cls._min(node.right)
        # go left and up while it's possible
        prev, cur = node, node.parent
        while cur is not None and cur.right is prev:
            prev, cur = cur, cur.parent
        return cur if cur is not node else None

    def _delete(self, node):
        if node is None:
            return False

        # is leaf or has one child
        if node.left is None or node.right is None:
            sub = (node.left if node.left is not None else node.right)
            if sub is not None:
                sub.parent = node.parent
            if node.parent is not None:
                if node.parent.left is node:
                    node.parent.left = sub
                elif node.parent.right is node:
                    node.parent.right = sub
                else:
                    raise RuntimeError("wrong parent reference")
            if node is self._root:
                self._root = sub
            if node.parent is not None:
                self._fix(node.parent)

        # has two children
        else:
            prev = self._prev(node)
            if prev is node:
                raise RuntimeError
            if prev is not node.left:
                prev.parent.right = prev.left
                if prev.left is not None:
                    prev.left.parent = prev.parent
                prev.left = node.left
                if node.left is not None:
                    node.left.parent = prev
            # whether prev is node.left or not
            prev.right = node.right
            if node.right is not None:
                node.right.parent = prev
            if node.parent is not None:
                if node.parent.left is node:
                    node.parent.left = prev
                elif node.parent.right is node:
                    node.parent.right = prev
            old_prev_parent = prev.parent
            prev.parent = node.parent
            if node is self._root:
                self._root = prev
            if old_prev_parent is not node:
                self._fix(old_prev_parent)
            self._fix(prev)
        return True

    def delete(self, number):
        return self._delete(self._find(key=number))

    @staticmethod
    def _merge_with_root(left, right, root):
        if root is None:
            raise ValueError
        else:
            root.left = left
            root.right = right
        if left is not None:
            left.parent = root
        if right is not None:
            right.parent = root
        return root

    @classmethod
    def _avl_merge_with_root(cls, left, right, root):
        if root is None:
            raise ValueError
        if left is None or right is None or abs(left.high - right.high) <= 1:
            cls._merge_with_root(left, right, root)
            cls._update_attrs(root)
            return root
        elif left.high > right.high:
            new_root = cls._avl_merge_with_root(left.right, right, root)
            left.right = new_root
            new_root.parent = left
            return cls._rebalance(left)
        else:
            new_root = cls._avl_merge_with_root(left, right.left, root)
            right.left = new_root
            new_root.parent = right
            return cls._rebalance(right)

    @classmethod
    def _merge(cls, left_root, right_root):
        if left_root is None and right_root is None:
            return None
        elif left_root is None:
            return right_root
        elif right_root is None:
            return left_root

        temp_tree = cls._from_root(left_root)
        new_root = cls._max(left_root)
        temp_tree._delete(new_root)
        new_root.parent = None
        left_root = temp_tree._root

        return cls._avl_merge_with_root(left_root, right_root, new_root)

    @classmethod
    def merge(cls, tree1, tree2):
        if (tree1._root is not None and tree2._root is not None and tree1.max() >= tree2.min()):
            raise ValueError(
                'every key of tree1 must be less than any key of tree2')
        return cls._from_root(cls._merge(tree1._root, tree2._root))

    @staticmethod
    def _clear_parent_attr(*nodes):
        for node in nodes:
            if node is not None:
                node.parent = None

    @classmethod
    def _split(cls, root, key):
        if root is None:
            return None, None
        if root.key > key:
            right_right = root.right
            left, right_left = cls._split(root.left, key)
            cls._clear_parent_attr(right_right, left, right_left)
            right = cls._avl_merge_with_root(right_left, right_right, root)
            cls._clear_parent_attr(right)
        else:
            left_left = root.left
            left_right, right = cls._split(root.right, key)
            cls._clear_parent_attr(left_left, left_right, right)
            left = cls._avl_merge_with_root(left_left, left_right, root)
            cls._clear_parent_attr(left)
        return left, right

    @classmethod
    def split(cls, tree, number):
        left_root, right_root = cls._split(tree._root, key=number)
        return cls._from_root(left_root), cls._from_root(right_root)

    def sum_between(self, left_num, right_num):
        left, temp = self._split(self._root, key=left_num - 1)
        middle, right = self._split(temp, key=right_num)
        res = middle.sum if middle is not None else 0
        temp = self._merge(middle, right)
        self._root = self._merge(left, temp)
        return res

    def _in_order_iter_by_iterative(self, node=None):
        cur = node if node is not None else self._root
        if cur is None:
            return
        visited = set()
        while True:
            if cur.left is not None and cur.left not in visited:
                cur = cur.left
            else:
                if cur not in visited:
                    yield cur
                    visited.add(cur)
                if cur.right is not None and cur.right not in visited:
                    cur = cur.right
                elif cur.parent is not None:
                    visited.discard(cur.left)
                    visited.discard(cur.right)
                    if cur is node:
                        return
                    cur = cur.parent
                else:
                    return

    def _reversed_in_order_iter_by_iterative(self, node=None):
        cur = node if node is not None else self._root
        if cur is None:
            return
        visited = set()
        while True:
            if cur.right is not None and cur.right not in visited:
                cur = cur.right
            else:
                if cur not in visited:
                    yield cur
                    visited.add(cur)
                if cur.left is not None and cur.left not in visited:
                    cur = cur.left
                elif cur.parent is not None:
                    visited.discard(cur.right)
                    visited.discard(cur.left)
                    if cur is node:
                        return
                    cur = cur.parent
                else:
                    return

    def _is_correct(self):
        def check_high(node):
            left_high = node.left.high if node.left is not None else -1
            right_high = node.right.high if node.right is not None else -1
            node_high = max(left_high, right_high) + 1
            return node.high == node_high

        def check_size(node):
            left_size = node.left.size if node.left is not None else 0
            right_size = node.right.size if node.right is not None else 0
            return node.size == left_size + right_size + 1

        def check_references(node):
            return not any([
                node.parent is node,
                node.left is not None and (
                    node.left is node or node.left.parent is not node),
                node.right is not None and (
                    node.right is node or node.right.parent is not node)
            ])

        if self._root is None:
            return True
        prev = None
        for cur in self._in_order_iter_by_iterative(node=self._root):
            if not check_high(cur):
                return False
            if not check_size(cur):
                return False
            if not check_references(cur):
                return False

            if not (
                (prev is None or prev.key < cur.key) and
                (cur.left is None or cur.left.key < cur.key) and
                (cur.right is None or cur.key < cur.right.key)
            ):
                return False
            prev = cur
        return True

    @staticmethod
    def _order_statistics(node, key):
        # counting from 1 is used in _order_statistics
        if key < 1:
            raise ValueError
        while node is not None:
            left_size = (node.left.size if node.left is not None else 0)
            if key == left_size + 1:
                return node
            elif key < left_size + 1:
                node = node.left
            else:
                node = node.right
                key -= left_size + 1
        return None

    def _slice(self, start, stop, step):
        def slice_iter(start_node, len_, step, next_el_func):
            cur_node = start_node
            for i, _ in enumerate(range(len_)):
                if cur_node is None:
                    break
                if i % step == 0:
                    yield cur_node.number
                cur_node = next_el_func(cur_node)
        start_node = self._order_statistics(self._root, start + 1)
        if start_node is None:
            return self.__class__()
        next_el_func = self._next if step > 0 else self._prev
        step = abs(step)
        return self.__class__(slice_iter(start_node, abs(stop - start), step, next_el_func))

    def min(self):
        return self._min(self._root).number if self._root is not None else None

    def max(self): 
        return self._max(self._root).number if self._root is not None else None

    def __len__(self):
        return self._root.size if self._root is not None else 0

    def __getitem__(self, key):
        if isinstance(key, int):
            if self._root is None:
                raise IndexError
            len_ = self._root.size
            if key >= len_ or key < -len_:
                raise IndexError
            key = (key + len_) % len_
            key += 1  # switch from 0-based to 1-based
            node = self._order_statistics(self._root, key)
            if node is None:
                raise RuntimeError
            return node.number
        elif isinstance(key, slice):
            if self._root is None:
                return self.__class__()
            start, stop, step = key.indices(self._root.size)
            return self._slice(start, stop, step)
        else:
            raise TypeError

    def __iter__(self):
        return (node.number for node in self._in_order_iter_by_iterative())

    def __reversed__(self):
        return (node.number for node in self._reversed_in_order_iter_by_iterative())


def make_queries(queries):
    s = 0
    def f(x): return (x + s) % 1000000001
    tree = AVLTree()
    result = []
    for action, *nums in queries:
        if action == "+":
            tree.add(f(nums[0]))
        elif action == "-":
            tree.delete(f(nums[0]))
        elif action == "?":
            result.append(tree.find(f(nums[0])))
        elif action == "s":
            a, b = map(f, nums)
            s = tree.sum_between(a, b)
            result.append(s)
    return result


def main():
    _ = stdin.readline()
    def convert(x): return int(x) if x.isdigit() else x
    queries = [list(map(convert, line.split())) for line in stdin]
    responses = make_queries(queries)
    for r in responses:
        if isinstance(r, bool):
            if r:
                print("Found")
            else:
                print("Not found")
        else:
            print(r)


def test():
    assert make_queries([
        ("?", 1),
        ("+", 1),
        ("?", 1),
        ("+", 2),
        ("s", 1, 2),
        ("+", 1000000000),
        ("?", 1000000000),
        ("-", 1000000000),
        ("?", 1000000000),
        ("s", 999999999, 1000000000),
        ("-", 2),
        ("?", 2),
        ("-", 0),
        ("+", 9),
        ("s", 0, 9)
    ]) == [False, True, 3, True, False, 1, False, 10]
    assert make_queries([
        ("?", 0),
        ("+", 0),
        ("?", 0),
        ("-", 0),
        ("?", 0)]
    ) == [False, True, False]
    assert make_queries([
        ("+", 491572259),
        ("?", 491572259),
        ("?", 899375874),
        ("s", 310971296, 877523306),
        ("+", 352411209)
    ]) == [True, False, 491572259]
    try:
        make_queries([
            ('+', 694485409), ('-', 617362355), ('s', 922995123, 973892015), ('-', 826907512), ('-', 63095135), ('+', 102168695), ('-', 710527848), ('+', 304971513), ('s', 771456177, 889070174), ('+', 136978966), ('?', 121195286), ('+', 659306257), ('+', 945676756), ('?', 418180746), ('?', 133374614), ('s', 370906274, 955987733), ('?', 106061030), ('?', 574852498), ('+', 222519398), ('?', 266127581), ('+', 734966233), ('+', 406409440), ('?', 532955017), ('-', 178779052), ('?', 388419607), ('-', 921746684), ('s', 548534536, 662862545), ('+', 246006933), ('-', 216256027), ('+', 841318453), ('s', 208342457, 381570162), ('?', 38264196), ('+', 547243860), ('?', 86102427), ('-', 454754428), ('?', 332689799), ('-', 363611680), ('+', 281495178), ('+', 769879165), ('s', 156578480, 981076501), ('s', 925219216, 951705759), ('+', 202114018), ('?', 123401863), ('s', 368300038, 854274434), ('s', 389692761, 848521372), ('+', 838090539), ('s', 149103819, 971919732), ('s', 692829419, 719212548), ('+', 833545309), ('s', 550862424, 929132063), ('s', 443127337, 821435821), ('+', 454907981), ('?', 657502213), ('?', 612353134), ('s', 307841051, 809193881), ('?', 325992747), ('-', 364115179), ('s', 848227772, 994755839), ('-', 294180846), ('+', 505933600), ('s', 710398640, 857285001), ('-', 647266402), ('-', 271114215), ('+', 362929531), ('-', 635577329), ('-', 184128365), ('s', 976265848, 976817743), ('-', 734005283), ('?', 790109629), ('?', 269723834), ('s', 247092383, 458058995), ('s', 743226871, 994665139), ('?', 780007439), ('+', 726618814), ('?', 457565076), ('-', 315977705), ('?', 140105971), ('-', 138541128), ('-', 715051248), ('-', 781474239), ('s', 520921396, 826503161), ('?', 762779592), ('s', 972010686, 995136978), ('-', 943515474), ('?', 678363740), ('s', 796535147, 805414554), ('?', 615447679), ('?', 682843239), ('?', 78747343), ('?', 969882719), ('s', 478925548, 520139717), ('?', 690671681), ('+', 774737718), ('-', 704906680), ('+', 318640857), ('-', 666207251), ('s', 567360945, 838054668), ('+', 561156351), ('s', 452273863, 961322441), ('-', 398207243), ('+', 667910292), ('s', 786269305, 886255555), ('?', 460333553), ('+', 956307258), ('+', 140486629), ('?', 286902161), ('-', 957959116), ('s', 535907869, 612791951), ('?', 676485786), ('-', 588635923), ('+', 647787127), ('?', 431945754), ('-', 282802995), ('?', 286238739), ('+', 83218552), ('s', 92450398, 473407406)])
    except RuntimeError:
        assert False
    try:
        make_queries([('?', 14), ('-', 13), ('?', 12), ('?', 9), ('?', 11), ('+', 12), ('s', 14, 15), ('s', 0, 0), ('+', 8), ('-', 15), ('s', 9, 14), ('-', 9), ('s', 3, 15), ('+', 12), ('s', 14, 14), ('-', 0), ('?', 15), ('s', 0, 4), ('?', 5), ('?', 15), ('-', 8), ('-', 1), ('s', 3, 9), ('-', 5), ('?', 11), ('-', 14), ('-', 5), ('-', 4), ('-', 9), ('-', 13), ('s', 12, 14), ('?', 13), ('-', 11), ('s', 4, 7), ('+', 2), ('+', 1), ('-', 14), ('?', 13), ('s', 11, 15), ('-', 5), ('-', 4), ('?', 13), ('+', 11), ('?', 10), ('?', 6), ('+', 5),
                      ('s', 6, 6), ('+', 0), ('-', 1), ('?', 6), ('?', 0), ('+', 12), ('-', 8), ('?', 9), ('s', 1, 7), ('+', 7), ('+', 7), ('+', 13), ('s', 6, 8), ('s', 2, 12), ('s', 7, 14), ('s', 14, 14), ('+', 8), ('+', 1), ('?', 9), ('+', 15), ('+', 13), ('+', 14), ('s', 5, 11), ('?', 8), ('?', 2), ('?', 7), ('s', 14, 15), ('s', 7, 12), ('?', 12), ('+', 3), ('+', 2), ('?', 15), ('+', 6), ('?', 7), ('s', 10, 14), ('s', 15, 15), ('+', 11), ('+', 3), ('s', 9, 12), ('+', 15), ('s', 14, 14), ('s', 14, 15), ('?', 14), ('+', 4), ('s', 4, 14)])
    except RuntimeError:
        assert False


def time_test(number=1, query_number=10 ** 5, min_num=0, max_num=10 ** 9):
    import gc
    import random
    import sys
    import timeit

    def prepare():
        actions = ("+", "-", "?", "s")
        queries = []
        for _ in range(query_number):
            action = random.choice(actions)
            a = random.randint(min_num, max_num)
            if action == "s":
                b = random.randint(a, max_num)
                q = (action, a, b)
            else:
                q = (action, a)
            queries.append(q)
        return queries

    print(
        timeit.timeit(
            stmt="gc.enable(); make_queries(prepare())",
            globals={**globals(), **locals()}, number=number) / number)


def time_test_worst_case(number=1, query_number=10 ** 5, min_num=0, max_num=10 ** 9):
    import gc
    import random
    import sys
    import timeit

    def prepare():
        queries = []
        action = "+"
        for _ in range(query_number // 2):
            a = random.randint(min_num, max_num)
            q = (action, a)
            queries.append(q)
        action = "s"
        for _ in range(query_number // 2):
            a = min_num
            b = max_num
            q = (action, a, b)
            queries.append(q)
        return queries

    print(
        timeit.timeit(
            stmt="gc.enable(); make_queries(prepare())",
            globals={**globals(), **locals()}, number=number) / number)


def test_tree_correctness(operations):
    print("===Test start===")
    performed = []
    tree = AVLTree()
    ops_funcs = {"add": tree.add,
                 "delete": tree.delete,
                 "min": tree.min,
                 "max": tree.max}
    curr_nums_in_tree = set()
    in_tree = set()
    not_in_tree = set()
    for op_name, *args in operations:
        performed.append((op_name, *args))
        op_func = ops_funcs[op_name]
        correct = None
        if op_name in ("add", "delete"):
            num = args[0]
            if op_name == "add":
                op_func(num)
                curr_nums_in_tree.add(num)
                in_tree.add(num)
                not_in_tree.discard(num)
                correct = (tree.find(num) == True)
                print(
                    f'Operation: {op_name}, number {num}; "{op_name}" correct: {correct}')
                assert correct, performed
            elif op_name == "delete":
                op_func(num)
                curr_nums_in_tree.discard(num)
                not_in_tree.add(num)
                in_tree.discard(num)
                correct = (tree.find(num) == False)
                print(
                    f'Operation: {op_name}, number {num}; "{op_name}" correct: {correct}')
                assert correct, performed
            correct = tree._is_correct()
            print(
                f"Operation: {op_name}, number {num}; tree correct: {correct}")
            assert correct, performed
        elif op_name in ("min", "max"):
            res = op_func()
            if op_name == "min":
                correct = (not curr_nums_in_tree or min(
                    curr_nums_in_tree) == res or tree._root is None)
            elif op_name == "max":
                correct = (not curr_nums_in_tree or max(
                    curr_nums_in_tree) == res or tree._root is None)
            print(f"Operation: {op_name}; {op_name} correct: {correct}")
            assert correct, performed

    print("\n===Check after all operations===")
    for num in in_tree:
        correct = (tree.find(num) == True)
        print(
            f"Left in tree num: {num}; there is the num in tree: {correct}")
        assert correct, performed
    for num in not_in_tree:
        correct = (tree.find(num) == False)
        print(
            f"Deleted from tree num: {num}; there is no the num in tree: {correct}")
        assert correct, performed
    print("===Test end===\n\n")


def tree_correctness_random_test(how_many_operations=100, op_names=("add", "delete", "min", "max")):
    import random

    def random_ops():
        for _ in range(how_many_operations):
            op_name = random.choice(op_names)
            args = []
            if op_name in ("add", "delete"):
                args.append(random.randint(min_num, max_num))
            yield (op_name, *args)
    cases = [
        (0, 10),
        (0, 100),
        (-10, 10),
        (-10 ** 5, 10 ** 5),
        (-2 ** 31, 2 ** 31 - 1)
    ]
    for i, case in enumerate(cases, 1):
        min_num, max_num = case
        test_tree_correctness(random_ops())
        padding = len(str(len(cases)))
        print(
            f"tree correctness random test for case {i:{padding}}/{len(cases)} finished.")
    print()


def tree_correctness_test_with_cases():
    cases = [
        [("add", 3), ("add", 1), ("delete", 3)],
        [("add", 10), ("add", 7), ("add", 9), ("add", 2), ("add", 6), ("delete", 6), ("delete",
                                                                                      2), ("add", 2), ("add", 3), ("add", 6), ("delete", 6), ("add", 1), ("delete", 2)],
        [("add", 87), ("add", 68), ("add", 71), ("delete", 11), ("delete", 59), ("delete", 1), ("add", 46), ("add", 6), ("delete", 100), ("add", 99), ("add", 92), ("delete", 94), ("add", 37), ("add", 56), ("delete", 82), ("delete", 27), ("delete", 68), ("delete", 57), ("add", 12), ("add", 8), ("delete", 99), ("add", 92), ("delete", 54), ("delete", 36), ("delete", 50), ("delete", 63), ("delete", 1), ("delete", 96), ("add", 41), ("add", 74), ("add", 42), ("add", 8), ("delete", 35), ("delete", 50), ("add", 93), ("delete", 6), ("delete", 31), ("delete", 58), ("delete", 76), ("delete", 8), ("delete",
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 25), ("add", 43), ("add", 1), ("delete", 63), ("delete", 75), ("delete", 71), ("add", 14), ("add", 48), ("add", 78), ("delete", 11), ("delete", 66), ("add", 84), ("add", 34), ("add", 38), ("delete", 41), ("add", 30), ("add", 75), ("add", 94), ("add", 16), ("delete", 78), ("delete", 85), ("delete", 10), ("delete", 64), ("delete", 24), ("delete", 41), ("add", 91), ("add", 16), ("delete", 2), ("add", 83), ("delete", 37), ("add", 67), ("delete", 46), ("add", 27), ("delete", 5), ("add", 19), ("add", 3), ("add", 72), ("delete", 66), ("delete", 1), ("add", 54), ("add", 51), ("delete", 51)],
        [('add', 6), ('add', 2), ('add', 7), ('add', 1), ('add', 0), ('delete', 2), ('add', 7), ('delete', 0), ('delete', 7), ('delete', 6), ('delete', 8), ('delete', 2), ('delete', 9), ('delete', 7), ('delete', 7), ('add', 1), ('delete', 5), ('delete', 10), ('delete', 10), ('add', 3), ('add', 2), ('add', 6), ('add', 3), ('add', 8), ('add', 1), ('add', 8), ('delete', 10), ('delete', 7), ('delete', 8), ('delete', 5), ('add', 8), ('delete', 7), ('delete', 9), ('add', 2), ('delete', 2), ('add', 2), ('add', 1), ('add', 3), ('add', 8), ('delete', 3), ('add', 7), ('add', 6), ('delete', 5), ('add', 7), ('delete', 0), ('delete', 8), ('delete', 10), ('add', 1), ('delete', 0), ('delete', 3), ('add', 0), ('delete', 8), ('delete', 6), ('delete', 7), ('delete', 2), ('add', 5), ('delete', 7), ('add', 10), ('add', 0), ('add', 7), ('delete', 3), ('add', 0), ('add', 10), ('add', 10), ('add', 5), ('delete', 5), ('add', 7), ('delete', 7), ('delete', 5), ('add', 10), ('delete', 6), ('delete', 2), ('add', 10), ('add', 5), ('delete', 8), ('delete', 9), ('add', 7), ('add', 6), ('delete', 4), ('delete', 1), ('delete', 1), ('delete', 7), ('add', 3), ('add', 10), ('add', 5), ('add', 0), ('delete', 3), ('delete', 10), ('add', 7), ('delete', 0), ('add', 10), ('add', 6), ('delete', 1), ('delete', 10), ('add', 9), ('add', 10), ('delete', 9), ('add', 4), ('delete', 3), ('delete', 9), ('add', 10), ('add', 3), ('delete', 4), ('delete', 4), ('delete', 0), ('add', 6), ('add', 2), ('add', 7), ('add', 4), ('delete', 5), ('delete', 0), ('delete', 1), ('delete', 8), ('add', 3), ('add', 2), ('delete', 2), ('add', 7), ('delete', 4), ('delete', 10), ('delete', 2), ('delete', 1), ('add', 9), ('add', 0), ('delete', 10), ('delete', 7), ('delete', 10), ('add', 2), ('delete', 4), ('add', 2), ('delete', 2), ('delete', 9), ('delete', 1), ('delete', 0), ('delete', 1), ('add', 4), ('delete', 4), ('delete', 1), ('add', 10), ('delete', 3), ('add', 2), ('delete', 2), ('delete', 6), ('delete', 10), ('add', 9), ('add', 0), ('add', 9), ('delete', 5), ('delete', 4), ('add', 0), ('delete', 1), ('delete', 5), ('add', 10),
         ('add', 7), ('delete', 9), ('add', 0), ('delete', 10), ('add', 1), ('delete', 1), ('add', 4), ('add', 7), ('add', 6), ('add', 4), ('add', 6), ('delete', 9), ('add', 3), ('add', 5), ('delete', 6), ('delete', 1), ('add', 6), ('add', 9), ('add', 7), ('delete', 10), ('add', 7), ('delete', 9), ('delete', 1), ('delete', 7), ('add', 0), ('delete', 7), ('add', 0), ('delete', 1), ('add', 9), ('delete', 6), ('delete', 9), ('delete', 2), ('delete', 7), ('delete', 9), ('add', 4), ('add', 5), ('add', 9), ('add', 0), ('delete', 2), ('delete', 2), ('delete', 7), ('add', 10), ('delete', 8), ('add', 1), ('delete', 1), ('delete', 5), ('add', 3), ('add', 9), ('delete', 10), ('delete', 1), ('add', 1), ('delete', 9), ('add', 8), ('add', 10), ('add', 3), ('delete', 1), ('add', 8), ('add', 2), ('add', 10), ('add', 8), ('add', 8), ('add', 6), ('add', 3), ('add', 3), ('delete', 9), ('delete', 7), ('add', 0), ('delete', 0), ('delete', 8), ('add', 6), ('delete', 6), ('add', 10), ('add', 1), ('add', 8), ('delete', 2), ('delete', 3), ('delete', 9), ('delete', 6), ('add', 6), ('add', 8), ('delete', 0), ('add', 3), ('delete', 2), ('delete', 4), ('delete', 0), ('add', 0), ('delete', 8), ('add', 8), ('add', 9), ('delete', 2), ('delete', 4), ('delete', 6), ('add', 1), ('delete', 1), ('delete', 10), ('add', 8), ('delete', 3), ('add', 6), ('delete', 1), ('add', 6), ('add', 10), ('add', 0), ('delete', 7), ('delete', 9), ('add', 9), ('add', 9), ('add', 1), ('add', 9), ('delete', 8), ('add', 7), ('delete', 2), ('delete', 3), ('delete', 0), ('add', 6), ('add', 8), ('add', 4), ('delete', 4), ('add', 10), ('delete', 1), ('delete', 2), ('delete', 3), ('delete', 9), ('delete', 1), ('delete', 1), ('delete', 4), ('delete', 6), ('delete', 1), ('delete', 2), ('add', 0), ('add', 5), ('delete', 0), ('delete', 0), ('add', 7), ('delete', 6), ('delete', 1), ('delete', 1), ('delete', 10), ('add', 7), ('delete', 8), ('add', 4), ('delete', 2), ('add', 9), ('add', 1), ('delete', 2), ('add', 7), ('delete', 6), ('delete', 5), ('add', 8), ('add', 6), ('delete', 6), ('delete', 9), ('delete', 4), ('add', 1), ('delete', 7)],
        [("add", 2), ("add", 1), ("add", 4), ("add", 0),
         ("add", 3), ("add", 6), ("add", 7), ("delete", 6)]
    ]
    for i, case in enumerate(cases, 1):
        padding = len(str(len(cases)))
        test_tree_correctness(case)
        print(
            f"tree correctness test for case {i:{padding}}/{len(cases)} finished.")


def test_merge(tree1_numbers, tree2_numbers):
    tree1_numbers = set(tree1_numbers)
    tree2_numbers = set(tree2_numbers)
    united_tree_numbers = sorted(tree1_numbers | tree2_numbers)
    tree1 = AVLTree(tree1_numbers)
    assert tree1._is_correct()
    tree2 = AVLTree(tree2_numbers)
    assert tree2._is_correct()
    united_tree = AVLTree.merge(tree1, tree2)
    assert united_tree._is_correct()
    united_tree_numbers_result = list(united_tree)
    assert united_tree_numbers == united_tree_numbers_result


def merge_random_test(how_many=1000):
    import random
    cases = [
        (10**2, 10**2, 0, 10*3, 10**5),
        (10**2, 10**2, 0, 10*3, 10**5),
        (10**2, 10**1, 0, 10*3, 10**5),
        (10**1, 10**2, 0, 10*3, 10**5),
        (10**2, 10**2, 0, 10**3, 10**5),
        (10**2, 10**2, 0, 10**3, 10**5),
        (10**2, 10**1, 0, 10**3, 10**5),
        (10**1, 10**2, 0, 10**3, 10**5)
    ]
    for i, case in enumerate(cases, 1):
        for j in range(1, how_many + 1):
            tree1_num_amount, tree2_num_amount, tree1_min_num, tree1_max_num, tree2_max_num = case
            tree1_numbers = set(random.randint(tree1_min_num, tree1_max_num)
                                for _ in range(tree1_num_amount))
            tree2_numbers = set(random.randint(
                tree1_max_num + 1, tree2_max_num) for _ in range(tree2_num_amount))
            test_merge(tree1_numbers, tree2_numbers)
            padding1, padding2 = len(str(how_many)), len(str(len(cases)))
            print(
                f"merge random test {j:{padding1}}/{how_many} for case {i:{padding2}}/{len(cases)} finished.", end='\r')
    print()


def merge_test_with_cases():
    cases = [
        ([0, 1, 2, 3, 4], [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]),
        ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 11, 12, 13, 14]),
        ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
          13, 14, 15, 16], [17, 18, 19, 20, 21]),
        ([], []),
        ([0], []),
        ([], [0]),
        ([0], [1]),
        ([0], [1]),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30], [1411, 1441, 1486, 2212, 2560, 4193, 7503, 8101, 8229, 9214, 10837, 11263, 12618, 13012, 13417, 15380, 15694, 16239, 16685, 17332, 17783, 18162, 18321, 19806, 20904, 23205, 23384, 24325, 25926, 26154, 29558, 29702, 31679, 31816, 34532, 39083, 40372, 40579, 40970, 41147, 41908, 42361, 43701,
                                                                                                                           44522, 48832, 50990, 52280, 53432, 53946, 54420, 54634, 55716, 56380, 57140, 58230, 58569, 59767, 60610, 61440, 61654, 61727, 63171, 64123, 64402, 66988, 68403, 70257, 71911, 72892, 73075, 73496, 74462, 75286, 76196, 77497, 79713, 79819, 79827, 80852, 81150, 81722, 81986, 83140, 85208, 85343, 85684, 86974, 87144, 89167, 90768, 93710, 93854, 94638, 94720, 96206, 97348, 97362, 97391, 97396, 99934])

    ]
    for i, case in enumerate(cases, 1):
        test_merge(*case)
        padding = len(str(len(cases)))
        print(
            f"merge test for case {i:{padding}}/{len(cases)} finished.", end='\r')
    print()


def test_split(numbers, key):
    from bisect import bisect_right
    numbers = sorted(set(numbers))
    key_occurs_end = bisect_right(numbers, key)
    tree1_nums, tree2_nums = numbers[:key_occurs_end], numbers[key_occurs_end:]
    tree = AVLTree(numbers)
    assert tree._is_correct()
    tree1, tree2 = AVLTree.split(tree, key)
    assert tree1._is_correct()
    assert tree2._is_correct()
    tree1_nums_res, tree2_nums_res = list(tree1), list(tree2)
    assert tree1_nums == tree1_nums_res and tree2_nums == tree2_nums_res


def split_random_test(how_many=1000):
    import random
    cases = [
        (0, 100, 1000)
    ]
    for i, case in enumerate(cases, 1):
        for j in range(1, how_many + 1):
            min_num, max_num, num_amount = case
            numbers = [random.randint(min_num, max_num)
                       for _ in range(num_amount)]
            key = random.choice(numbers)
            test_split(numbers, key)
            padding1, padding2 = len(str(how_many)), len(str(len(cases)))
            print(
                f"split random test {j:{padding1}}/{how_many} for case {i:{padding2}}/{len(cases)} finished.", end='\r')
    print()


def split_test_with_cases():
    cases = [
        ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 5),
        ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
          52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100], 79),
        ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
          52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100], 54),
        ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
          52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100], 49),
        ([1, 2, 3], 0),
        ([1, 2, 3], 3),
        ([], 1)
    ]
    for i, case in enumerate(cases, 1):
        test_split(*case)
        padding = len(str(len(cases)))
        print(
            f"split test for case {i:{padding}}/{len(cases)} finished.", end='\r')
    print()


def slice_test_with_cases():
    tree = AVLTree([1, 2, 3, 4, 5])
    assert tree._is_correct()
    cases = [
        (tree[:], [1, 2, 3, 4, 5]),
        (tree[1:], [2, 3, 4, 5]),
        (tree[:-1], [1, 2, 3, 4]),
        (tree[::-2], [1, 3, 5]),
        (tree[::-1], [1, 2, 3, 4, 5]),
        (tree[2:-2], [3]),
        (tree[-2:2:-1], [4]),
        (tree[6:12], []),
        (tree[6:12:-1], []),
        (tree[3:55], [4, 5]),
        (tree[-1:-4:-1], [3, 4, 5]),
        (AVLTree()[1:25], [])
    ]
    for i, case in enumerate(cases, 1):
        tested_tree, result_list = case
        assert tested_tree._is_correct()
        assert list(tested_tree) == result_list
        padding = len(str(len(cases)))
        print(
            f"slice test for case {i:{padding}}/{len(cases)} finished.", end='\r')
    print()


if __name__ == "__main__":
    # main()
    test()
    # # time_test(10)
    # # time_test_worst_case(1)
    # time_test(10 ** 5, query_number=100, min_num=0, max_num=15)
    tree_correctness_test_with_cases()
    tree_correctness_random_test(1000)
    merge_test_with_cases()
    merge_random_test(1000)
    split_test_with_cases()
    split_random_test(1000)
    slice_test_with_cases()
