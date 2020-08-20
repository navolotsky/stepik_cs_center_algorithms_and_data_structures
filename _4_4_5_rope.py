import random
import string
import timeit
from sys import stdin


class Node:
    def __init__(self, value):
        self.value = value
        self.parent = None
        self.left = None
        self.right = None
        self.size = 1

    def __repr__(self):
        return "{}<{}, {}, has: parent {}, left {}, right {}>".format(
            self.__class__.__name__, repr(self.value), self.size,
            self.parent is not None,
            self.left is not None, self.right is not None)

    def __str__(self):
        return "{}({})".format(self.__class__.__name__, repr(self.value))

    def __hash__(self):
        return id(self)


class SplayTree:
    """Indexed Splay tree implementation.
    Splay tree is self-balancing binary search tree
    having logarithmic height on average case.

    This implementation use subtree size as a key, so it behaves exactly
    like the built-in list in part of methods of use.

    Single element operations: takes O(log(n)) time on average case
    to indexing (-), assigment (-), deleting (+) from any place,
    or inserting (+) at any place.

    Slice operations: takes O(log(n) + m) time on average case
    to getting (-), deleting (+) and O(log(n) + m + k) time
    on average case to setting (+).

    Legend:
    n is the number of nodes in the tree;
    m is the number of nodes in the requested slice of the tree;
    k is the number of nodes in the given iterable;
    (-) marks an operation that likely works slower than
    in the built-in list on average case with big input;
    (+) marks an operation that likely works faster than
    in the built-in list on average case with big input.
    """

    def __init__(self, values=None):
        self._root = None
        if values is None:
            return
        last_node = None
        for value in values:
            node = Node(value)
            if self._root is None:
                self._root = node
            else:
                if last_node is None:
                    last_node = self._root
                last_node.right = node
                node.parent = last_node
                self._splay(node)
            last_node = node

    def __repr__(self):
        return "{}<root: {}, size: {}>".format(
            self.__class__.__name__, repr(self._root), self.__len__())

    def __str__(self):
        return "{}({})".format(self.__class__.__name__, repr(self._root))

    @classmethod
    def _from_root(cls, root):
        """Return a tree builded from a given node considering as a tree root.

        The "parent" attribute of a given "root" must be None.
        """
        if root is not None and root.parent is not None:
            raise ValueError('"parent" is not None')
        tree = cls()
        tree._root = root
        return tree

    @staticmethod
    def _update_attrs(*nodes):
        """Update attributes given nodes using their direct children."""
        # Notice:
        # The method is in this class and not in class Node
        # because SplayTree affects the attributes of Node instances
        # by changing deep children in their subtrees, not only direct
        # children.
        # Also class Node is somewhat data class.
        for node in nodes:
            if node is None:
                continue
            # update size
            left_size = node.left.size if node.left is not None else 0
            right_size = node.right.size if node.right is not None else 0
            node.size = left_size + right_size + 1

    def _splay(self, node):
        """Make the given "node" the tree root.

        The defining operation of a Splay tree.
        """
        if node is None:
            raise ValueError
        while node.parent is not None:
            old_father = node.parent

            old_grandfather = old_father.parent
            old_great_grandfather = None
            if old_grandfather is not None:
                old_great_grandfather = old_grandfather.parent

            if old_father.left is node:
                # node is left son and has no grandfather (zig)
                if old_grandfather is None:
                    old_father.left = node.right
                    if node.right is not None:
                        node.right.parent = old_father

                    node.right = old_father
                    old_father.parent = node

                # node is left son
                # and father is left son of grandfather (zigzig)
                elif old_grandfather.left is old_father:
                    old_father.left = node.right
                    if node.right is not None:
                        node.right.parent = old_father

                    old_grandfather.left = old_father.right
                    if old_father.right is not None:
                        old_father.right.parent = old_grandfather

                    node.right = old_father
                    old_father.parent = node

                    old_father.right = old_grandfather
                    old_grandfather.parent = old_father

                # node is left son
                # and father is right son of grandfather (zigzag)
                elif old_grandfather.right is old_father:
                    old_father.left = node.right
                    if node.right is not None:
                        node.right.parent = old_father

                    old_grandfather.right = node.left
                    if node.left is not None:
                        node.left.parent = old_grandfather

                    node.right = old_father
                    old_father.parent = node

                    node.left = old_grandfather
                    old_grandfather.parent = node
                else:
                    raise RuntimeError

            elif old_father.right is node:
                # node is right son and has no grandfather (zag)
                if old_grandfather is None:
                    old_father.right = node.left
                    if node.left is not None:
                        node.left.parent = old_father

                    node.left = old_father
                    old_father.parent = node

                # node is right son
                # and father is right son of grandfather (zagzag)
                elif old_grandfather.right is old_father:
                    old_father.right = node.left
                    if node.left is not None:
                        node.left.parent = old_father

                    old_grandfather.right = old_father.left
                    if old_father.left is not None:
                        old_father.left.parent = old_grandfather

                    node.left = old_father
                    old_father.parent = node

                    old_father.left = old_grandfather
                    old_grandfather.parent = old_father

                # node is right son
                # and father is left son of grandfather (zagzig)
                elif old_grandfather.left is old_father:
                    old_father.right = node.left
                    if node.left is not None:
                        node.left.parent = old_father

                    old_grandfather.left = node.right
                    if node.right is not None:
                        node.right.parent = old_grandfather

                    node.left = old_father
                    old_father.parent = node

                    node.right = old_grandfather
                    old_grandfather.parent = node
                else:
                    raise RuntimeError
            else:
                raise RuntimeError

            node.parent = old_great_grandfather
            if old_great_grandfather is not None:
                if old_great_grandfather.left is old_grandfather:
                    old_great_grandfather.left = node
                elif old_great_grandfather.right is old_grandfather:
                    old_great_grandfather.right = node
                else:
                    raise RuntimeError
            self._update_attrs(
                old_grandfather, old_father, node, old_great_grandfather)

        self._root = node

    @staticmethod
    def _first(node):
        prev, cur = node, node.left
        while cur is not None:
            prev, cur = cur, cur.left
        return prev

    @staticmethod
    def _last(node):
        prev, cur = node, node.right
        while cur is not None:
            prev, cur = cur, cur.right
        return prev

    @classmethod
    def _prev(cls, node):
        # go right and down in left subtree
        if node.left is not None:
            return cls._last(node.left)
        # go right and up while it's possible
        prev, cur = node, node.parent
        while cur is not None and cur.left is prev:
            prev, cur = cur, cur.parent
        return cur if cur is not node else None

    @classmethod
    def _next(cls, node):
        # go left and down in right subtree
        if node.right is not None:
            return cls._first(node.right)
        # go left and up while it's possible
        prev, cur = node, node.parent
        while cur is not None and cur.right is prev:
            prev, cur = cur, cur.parent
        return cur if cur is not node else None

    def _append(self, node):
        if self._root is None:
            self._root = node
            return
        last_node = self._last(self._root)
        last_node.right = node
        node.parent = last_node
        self._splay(node)

    def append(self, value):
        self._append(Node(value))

    @staticmethod
    def _order_statistics(node, key):
        """Find the k-order statistics. Numbering starts with one."""
        if key < 1:
            raise ValueError(
                'The "key" must be positive integer (1-based numbering).')
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

    def _insert_before(self, node, new_node):
        prev_node = self._prev(node)
        if prev_node is None:
            node.left = new_node
            new_node.parent = node
        else:
            prev_old_right = prev_node.right
            prev_node.right = new_node
            new_node.parent = prev_node
            if prev_old_right is not None:
                new_node.right = prev_old_right
                prev_old_right.parent = new_node
        self._update_attrs(new_node, prev_node, node)
        self._splay(new_node)

    def _insert(self, index, node):
        if self._root is None:
            self._root = node
            return
        if index >= self._root.size:
            self._append(node)
            return
        if index < -self._root.size:
            index = 0
        elif index < 0:
            index %= self._root.size
        cur_by_ind = self._order_statistics(self._root, index + 1)
        self._insert_before(cur_by_ind, node)

    def insert(self, index, value):
        """Insert object before index.

        Behaves like built-in list.insert."""
        self._insert(index, Node(value))

    def index(self, value, start=0, stop=9223372036854775807):
        """Return first index of value.

        Raises ValueError if the value is not present.
        Behaves like built-in list.index().
        """
        if (self._root is not None and
                start < self._root.size and
                stop >= -self._root.size):
            # Start from 0 due to need to count index of given value.
            # If to convert "start" to positive without doing that,
            # value of "start" may be incorrect.
            if start < -self._root.size:
                start = 0
            # Convert start and stop to positive.
            start %= self._root.size
            # Exclude case "stop %= 1"
            # else "start" will be equal to "stop"
            # when it's actually not.
            if stop < 0:
                stop %= self._root.size
            start_node = self._order_statistics(self._root, start + 1)
            if start_node is not None:
                for i, node in enumerate(
                        self._in_order_iter(start_node),
                        start):
                    if i >= stop:
                        break
                    if node.value == value:
                        return i
        raise ValueError(f"{value} is not in {self.__class__.__name__}")

    @classmethod
    def _merge(cls, left_root, right_root):
        if left_root is None and right_root is None:
            return None
        elif left_root is None:
            return right_root
        elif right_root is None:
            return left_root
        new_root = cls._last(left_root)
        temp_tree = cls._from_root(left_root)
        temp_tree._splay(new_root)
        new_root.right = right_root
        right_root.parent = new_root
        temp_tree._update_attrs(new_root)
        return new_root

    @classmethod
    def merge(cls, tree1, tree2):
        return cls._from_root(cls._merge(tree1._root, tree2._root))

    def _delete(self, node):
        self._splay(node)
        if node.left is not None:
            node.left.parent = None
        if node.right is not None:
            node.right.parent = None
        self._root = self._merge(node.left, node.right)

    @classmethod
    def _split(cls, root, key):
        """Return root[0:key], root[key:]"""
        if root is None:
            return None, None
        if key < -root.size:
            return None, root
        if key >= root.size:
            return root, None
        key %= root.size
        node = cls._order_statistics(root, key + 1)
        temp_tree = cls._from_root(root)
        temp_tree._splay(node)
        old_node_left = node.left
        if old_node_left is not None:
            node.left = None
            old_node_left.parent = None
            temp_tree._update_attrs(node)
        return old_node_left, node

    @classmethod
    def split(cls, tree, index):
        """Split given "tree" into two trees by given "index".
        Values starting with given index will be in second tree."""
        left_root, right_root = cls._split(tree._root, key=index)
        return cls._from_root(left_root), cls._from_root(right_root)

    def _in_order_iter(self, start_node=None, end_node=None, root_node=None):
        """Return in-order iterator of tree nodes from start_node
        to end_node including both. "root_node" limits iteration
        with subtree in given node.
        """
        cur = root_node if root_node is not None else self._root
        if cur is None:
            return
        visited = set()
        while True:
            if (cur.left is not None and cur.left not in visited and
                    cur is not start_node):
                cur = cur.left
            else:
                if cur not in visited:
                    yield cur
                    visited.add(cur)
                if (cur.right is not None and cur.right not in visited and
                        cur is not end_node):
                    cur = cur.right
                elif cur.parent is not None:
                    visited.discard(cur.left)
                    visited.discard(cur.right)
                    if cur is root_node:
                        return
                    cur = cur.parent
                else:
                    return

    def _reversed_in_order_iter(
            self, start_node=None, end_node=None, root_node=None):
        cur = root_node if root_node is not None else self._root
        if cur is None:
            return
        visited = set()
        while True:
            if (cur.right is not None and cur.right not in visited and
                    cur is not start_node):
                cur = cur.right
            else:
                if cur not in visited:
                    yield cur
                    visited.add(cur)
                if (cur.left is not None and cur.left not in visited and
                        cur is not end_node):
                    cur = cur.left
                elif cur.parent is not None:
                    visited.discard(cur.right)
                    visited.discard(cur.left)
                    if cur is root_node:
                        return
                    cur = cur.parent
                else:
                    return

    def __len__(self):
        return self._root.size if self._root is not None else 0

    def __getitem__(self, key):
        if isinstance(key, int):
            if self._root is None:
                raise IndexError
            len_ = self._root.size
            if key >= len_ or key < -len_:
                raise IndexError
            # convert "key" to positive and add 1
            # to meet _order_statistics()'s constraints on "key"
            k = key % self._root.size + 1
            node = self._order_statistics(self._root, k)
            self._splay(node)
            return node.value
        elif isinstance(key, slice):
            if self._root is None:
                return self.__class__()
            indices = key.indices(self._root.size)
            start, stop, step = indices
            if ((step > 0 and start >= stop) or
                    (step < 0 and start <= stop)):
                return self.__class__()
            if start > stop and step < 0:
                split_key = stop + 1
            else:
                split_key = start
            if split_key < 0:
                split_key %= self._root.size
            left, temp = self._split(self._root, split_key)
            len_ = abs(stop - start)
            requested, right = self._split(temp, len_)
            source = self._from_root(requested)
            node_it = (source._in_order_iter() if step > 0
                       else source._reversed_in_order_iter())
            step_pos = abs(step)
            if step_pos > 1:
                val_it = (node.value for i, node in enumerate(
                    node_it) if i % step_pos == 0)
            else:
                val_it = (node.value for node in node_it)
            result = self.__class__(val_it)
            temp = self._merge(requested, right)
            self._root = self._merge(left, temp)
            return result
        else:
            raise TypeError

    def __delitem__(self, key):
        if isinstance(key, int):
            if self._root is None:
                raise IndexError
            len_ = self._root.size
            if key >= len_ or key < -len_:
                raise IndexError
            k = key % self._root.size + 1
            node = self._order_statistics(self._root, k)
            self._delete(node)
        elif isinstance(key, slice):
            if key.step == 0:
                raise ValueError("slice step cannot be zero")
            if self._root is None:
                return  # tree is empty
            indices = key.indices(self._root.size)
            start, stop, step = indices
            if ((step > 0 and start >= stop) or
                    (step < 0 and start <= stop)):
                return  # request to delete empty slice
            if start > stop and step < 0:
                split_key = stop + 1
            else:
                split_key = start
            if split_key < 0:
                split_key %= self._root.size
            left, temp = self._split(self._root, split_key)
            len_ = abs(stop - start)
            requested, right = self._split(temp, len_)
            step_pos = abs(step)
            if step_pos != 1 and requested is not None:
                source = self._from_root(requested)
                if step > 0:
                    next_el_func = self._next
                    start_node = self._first(source._root)
                else:
                    next_el_func = self._prev
                    start_node = self._last(source._root)
                prev, cur = None, start_node
                for i in range(len_):
                    prev, cur = cur, next_el_func(cur)
                    if i % step_pos == 0:
                        source._delete(prev)
                temp = self._merge(source._root, right)
            else:
                temp = right
            self._root = self._merge(left, temp)
        else:
            raise TypeError

    def __setitem__(self, key, value):
        if isinstance(key, int):
            if self._root is None:
                raise IndexError
            len_ = self._root.size
            if key >= len_ or key < -len_:
                raise IndexError
            k = key % self._root.size + 1
            node = self._order_statistics(self._root, k)
            node.value = value
            self._splay(node)
        elif isinstance(key, slice):
            if key.step == 0:
                raise ValueError("slice step cannot be zero")
            try:
                val_it = iter(value)
            except TypeError:
                if key.step is None or key.step == 1:
                    msg = "can only assign an iterable"
                else:
                    msg = "must assign iterable to extended slice"
                raise TypeError(msg)

            # Case when self is empty:
            if self._root is None:
                if key.step is None or key.step == 1:
                    self._root = self.__class__(val_it)._root
                    return
                vals = list(val_it)
                if vals:
                    raise ValueError(
                        f"attempt to assign sequence of size {len(vals)}"
                        " to extended slice of size 0")
                return

            # Get subtree related to the slice.
            indices = key.indices(self.__len__())
            start, stop, step = indices
            if start > stop and step < 0:
                split_key = stop + 1
            else:
                split_key = start
            if split_key < 0:
                split_key %= self._root.size
            left, temp = self._split(self._root, split_key)
            if step > 0 and start >= stop or step < 0 and start <= stop:
                len_ = 0
            else:
                len_ = abs(stop - start)
            requested, right = self._split(temp, len_)

            # Case when simple replaceable slice:
            if step == 1:
                result = self.__class__(val_it)._root
                temp = self._merge(result, right)
                self._root = self._merge(left, temp)
                return

            # Case when extended (step != 1) replaceable slice:
            step_pos = abs(step)
            req_size = requested.size if requested is not None else 0
            slice_len = (req_size // step_pos +
                         (1 if req_size % step_pos != 0 else 0))
            vals = list(val_it)
            if step != 1 and slice_len != len(vals):
                temp = self._merge(requested, right)
                self._root = self._merge(left, temp)
                raise ValueError(
                    f"attempt to assign sequence of size {len(vals)}"
                    " to extended slice of size {slice_len}")
            source = self._from_root(requested)
            node_it = (source._in_order_iter() if step > 0
                       else source._reversed_in_order_iter())
            if step_pos > 1:
                edited_nodes_it = (node for i, node in enumerate(
                    node_it) if i % step_pos == 0)
            else:
                edited_nodes_it = node_it
            for edited_node, new_value in zip(edited_nodes_it, vals):
                edited_node.value = new_value
            temp = self._merge(requested, right)
            self._root = self._merge(left, temp)
        else:
            raise TypeError

    def __iter__(self):
        return (node.value for node in self._in_order_iter())

    def __reversed__(self):
        return (node.value for node in self._reversed_in_order_iter())

    def _split_into_three_subtrees(self, slice_):
        indices = slice_.indices(self.__len__())
        start, stop, step = indices
        if start > stop and step < 0:
            split_key = stop + 1
        else:
            split_key = start
        if split_key < 0:
            split_key %= self._root.size
        left, temp = self._split(self._root, split_key)
        if step > 0 and start >= stop or step < 0 and start <= stop:
            len_ = 0
        else:
            len_ = abs(stop - start)
        requested, right = self._split(temp, len_)
        return left, requested, right

    def slices_substitution(self, src_start, src_stop, dest_start, dest_stop):
        left, source, right = self._split_into_three_subtrees(
            slice(src_start, src_stop, 1))
        self._root = self._merge(left, right)
        req_size = source.size if source is not None else 0
        dest_start_changed = dest_start - req_size
        dest_stop_changed = dest_stop - req_size
        left, _, right = self._split_into_three_subtrees(
            slice(dest_start_changed, dest_stop_changed, 1))
        temp = self._merge(source, right)
        self._root = self._merge(left, temp)


def check_whether_tree_is_correct(tree):
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

    if tree._root is None:
        return True
    for cur in tree._in_order_iter():
        if not check_size(cur):
            return False
        if not check_references(cur):
            return False
    return True


def make_queries(source_string, queries):
    tree = SplayTree(source_string)

    # too slow:
    # for i, j, k in queries:
    #     from_start, from_stop, to = i, j + 1, k
    #     subtree = tree[from_start:from_stop]
    #     del tree[from_start:from_stop]
    #     tree[to:to] = subtree

    # obviously much faster:
    for i, j, k in queries:
        source_start, source_stop = i, j + 1
        dest_start = dest_stop = k + (j - i + 1)
        tree.slices_substitution(
            source_start, source_stop, dest_start, dest_stop)
    return ''.join(tree)


def main():
    source_string = stdin.readline().rstrip()
    _ = stdin.readline()
    queries = [list(map(int, line.split())) for line in stdin]
    resulting_string = make_queries(source_string, queries)
    print(resulting_string)


def test():
    assert make_queries(
        "hlelowrold",
        [(1, 1, 2),
         (6, 6, 7)]
    ) == "helloworld"
    assert make_queries(
        "abcdef",
        [(0, 1, 1),
         (4, 5, 0)]
    ) == "efcabd"
    assert make_queries(
        "",
        [(0, 0, 0)]
    ) == ""
    assert make_queries(
        "a",
        [(0, 0, 0)]
    ) == "a"
    assert make_queries(
        "abcde",
        [(0, 4, 0)]
    ) == "abcde"


def time_test(number=1, query_number=10 ** 5, tree_size=3 * 10 ** 5):
    def prepare():  # pylint: disable=possibly-unused-variable
        values = [random.choice(string.ascii_lowercase)
                  for _ in range(tree_size)]
        queries = []
        for _ in range(query_number):
            i = random.randint(0, len(values))
            j = random.randint(i, len(values))
            k = random.randint(0, len(values) - (j - i))
            queries.append((i, j, k))
        return values, queries

    print(
        timeit.timeit(
            stmt="import gc; gc.enable(); make_queries(*prepare())",
            globals={**globals(), **locals()}, number=number) / number)


def test_merge(tree1_values, tree2_values):
    united_tree_values = tree1_values + tree2_values
    tree1 = SplayTree(tree1_values)
    assert check_whether_tree_is_correct(tree1)
    tree2 = SplayTree(tree2_values)
    assert check_whether_tree_is_correct(tree2)
    united_tree = SplayTree.merge(tree1, tree2)
    assert check_whether_tree_is_correct(united_tree)
    united_tree_values_result = list(united_tree)
    assert united_tree_values == united_tree_values_result


def merge_random_test(how_many=1000):
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
            (
                tree1_num_amount, tree2_num_amount,
                tree1_min_num, tree1_max_num, tree2_max_num
            ) = case
            tree1_vals = list(random.randint(tree1_min_num, tree1_max_num)
                              for _ in range(tree1_num_amount))
            tree2_vals = list(random.randint(tree1_max_num + 1, tree2_max_num)
                              for _ in range(tree2_num_amount))
            test_merge(tree1_vals, tree2_vals)
            padding1, padding2 = len(str(how_many)), len(str(len(cases)))
            print(f"merge random test {j:{padding1}}/{how_many} "
                  f"for case {i:{padding2}}/{len(cases)} finished.", end='\r')
    print()


def merge_test_with_cases():
    cases = [
        (
            [0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]),
        ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 11, 12, 13, 14]),
        ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
          13, 14, 15, 16], [17, 18, 19, 20, 21]),
        ([], []),
        ([0], []),
        ([], [0]),
        ([0], [1]),
        ([0], [1]),
        (
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
             20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
            [1411, 1441, 1486, 2212, 2560, 4193, 7503, 8101, 8229, 9214, 10837,
             11263, 12618, 13012, 13417, 15380, 15694, 16239, 16685, 17332,
             17783, 18162, 18321, 19806, 20904, 23205, 23384, 24325, 25926,
             26154, 29558, 29702, 31679, 31816, 34532, 39083, 40372, 40579,
             40970, 41147, 41908, 42361, 43701, 44522, 48832, 50990, 52280,
             53432, 53946, 54420, 54634, 55716, 56380, 57140, 58230, 58569,
             59767, 60610, 61440, 61654, 61727, 63171, 64123, 64402, 66988,
             68403, 70257, 71911, 72892, 73075, 73496, 74462, 75286, 76196,
             77497, 79713, 79819, 79827, 80852, 81150, 81722, 81986, 83140,
             85208, 85343, 85684, 86974, 87144, 89167, 90768, 93710, 93854,
             94638, 94720, 96206, 97348, 97362, 97391, 97396, 99934])

    ]
    for i, case in enumerate(cases, 1):
        test_merge(*case)
        padding = len(str(len(cases)))
        print(f"merge test for case {i:{padding}}/{len(cases)} finished.",
              end='\r')
    print()


def test_split(values, index):
    tree1_nums, tree2_nums = values[:index], values[index:]
    tree = SplayTree(values)
    assert check_whether_tree_is_correct(tree)
    tree1, tree2 = SplayTree.split(tree, index)
    assert check_whether_tree_is_correct(tree1)
    assert check_whether_tree_is_correct(tree2)
    tree1_nums_res, tree2_nums_res = list(tree1), list(tree2)
    assert tree1_nums == tree1_nums_res and tree2_nums == tree2_nums_res


def split_test_with_cases():
    cases = [
        ([], 0),
        (['a'], 0),
        (['a'], -1),
        (['a', 'b'], 1),
        (['a', 'b'], -100),
        (['a', 'b'], 100)
    ]
    for i, case in enumerate(cases, 1):
        test_split(*case)
        padding = len(str(len(cases)))
        print(f"split test for case {i:{padding}}/{len(cases)} finished.",
              end='\r')
    print()


def split_random_test(how_many=1000):
    cases = [
        (0, 100, 1000)
    ]
    for i, case in enumerate(cases, 1):
        for j in range(1, how_many + 1):
            min_num, max_num, num_amount = case
            values = [random.randint(min_num, max_num)
                      for _ in range(num_amount)]
            index = random.randint(- 2 * len(values), 2 * len(values))
            test_split(values, index)
            padding1, padding2 = len(str(how_many)), len(str(len(cases)))
            print(f"split random test {j:{padding1}}/{how_many} "
                  f"for case {i:{padding2}}/{len(cases)} finished.", end='\r')
    print()


def test_append(values):
    tree = SplayTree()
    for value in values:
        tree.append(value)
    tree_vals = list(tree)
    assert values == tree_vals


def append_test_with_cases():
    cases = [
        [],
        [1],
        [1, 2, 3],
        [3, 2, 1],
        ['a', 'b', 'c'],
        ['c', 'b', 'a'],
        ['c', 'a', 'c'],
        ['aa', 'ab', 'bb'],
        ['ab', 'bb', 'aa']
    ]
    for i, case in enumerate(cases, 1):
        test_append(case)
        padding = len(str(len(cases)))
        print(f"append test for case {i:{padding}}/{len(cases)} finished.",
              end='\r')
    print()


def append_random_test(how_many=1000):
    cases = [
        (0, 100, 1000)
    ]
    for i, case in enumerate(cases, 1):
        for j in range(1, how_many + 1):
            min_num, max_num, num_amount = case
            values = [random.randint(min_num, max_num)
                      for _ in range(num_amount)]
            test_append(values)
            padding1, padding2 = len(str(how_many)), len(str(len(cases)))
            print(f"append random test {j:{padding1}}/{how_many} "
                  f"for case {i:{padding2}}/{len(cases)} finished.", end='\r')
    print()


def test_index(values):
    tree = SplayTree(values)
    for val in values:
        assert tree.index(val) == values.index(val)


def index_test_with_cases():
    cases = [
        [],
        [1],
        [1, 2, 3],
        [3, 2, 1],
        ['a', 'b', 'c'],
        ['c', 'b', 'a'],
        ['c', 'a', 'c'],
        ['aa', 'ab', 'bb'],
        ['ab', 'bb', 'aa']
    ]
    for i, case in enumerate(cases, 1):
        test_index(case)
        padding = len(str(len(cases)))
        print(f"index test for case {i:{padding}}/{len(cases)} finished.",
              end='\r')
    print()


def index_random_test(how_many=1000):
    cases = [
        (0, 100, 1000)
    ]
    for i, case in enumerate(cases, 1):
        for j in range(1, how_many + 1):
            min_num, max_num, num_amount = case
            values = [random.randint(min_num, max_num)
                      for _ in range(num_amount)]
            test_index(values)
            padding1, padding2 = len(str(how_many)), len(str(len(cases)))
            print(f"index random test {j:{padding1}}/{how_many} "
                  f"for case {i:{padding2}}/{len(cases)} finished.", end='\r')
    print()


def index_test_with_special_cases():
    cases0 = [
        ([], ['a']),
        (['a', 'b'], ['a', 2]),
        (['a', 'b'], ['a', 0, -3]),
        (['a'], ['a', 0, -1]),
        (['a', 'b'], ['a', 0, 0]),
        (['a', 'b'], ['b', -1, -1]),
        (['a', 'b'], ['a', 0, -2]),
        (['a', 'b'], ['b', 1, -1]),
        (['a', 'b'], ['b', -1, 1])
    ]
    for case in cases0:
        values, index_args = case
        try:
            SplayTree(values).index(*index_args)
        except ValueError:
            assert True
        else:
            assert False, case
    cases1 = [
        (['a'], ['a', 0], 0),
        (['a'], ['a', 0, 1], 0),
        (['a', 'b'], ['b', -1], 1),
        (['a', 'b'], ['a', -2, -1], 0),
        (['a', 'b'], ['a', -5, -1], 0),
        (['a', 'b', 'c'], ['b', -55555], 1),
        (['a', 'b', 'c'], ['b', 0, 55555], 1)
    ]
    for case in cases1:
        values, index_args, res = case
        assert res == SplayTree(values).index(*index_args)


def insert_test_with_special_cases():
    cases = [
        ([], 'a', 98, 0),
        (['a', 'b', 'c'], 'd', -1000, 0),
        (['a', 'b', 'c'], 'd', 1000, 3),
        (['a', 'b', 'c'], 'd', 0, 0),
        (['a', 'b', 'c'], 'd', -1, 2),
        (['a', 'b', 'c'], 'd', 2, 2),
        (['a', 'b', 'c'], 'd', 3, 3),
        (['a', 'b', 'c'], 'd', 1, 1),
    ]
    for case in cases:
        values, new_value, ins_index, res_index = case
        tree = SplayTree(values)
        tree.insert(ins_index, new_value)
        assert tree[res_index] == new_value


def rnd_int(values_len):
    return random.randint(- (2 * values_len + 1), 2 * values_len + 1)


def rnd_nonzero_int(values_len):
    res = 0
    while res == 0:
        res = random.randint(- (2 * values_len + 1), 2 * values_len + 1)
    return res


def gen_cases_for_getitem_or_delitem(
        how_many=1000, keys_maxlen=5, values_maxlen=10, values_source=None):

    if values_source is None:
        values_source = string.ascii_lowercase
    values_source = sorted(set(values_source))
    if len(values_source) < values_maxlen:
        raise ValueError(
            '"values_source" must contain a number of unique values '
            'greater than or equal to "values_maxlen"')
    for _ in range(how_many):
        values_len = random.randint(0, values_maxlen)
        keys_len = random.randint(1, keys_maxlen)
        values = values_source[:values_maxlen]
        keys = []
        for _ in range(keys_len):
            type_ = random.choice([int, slice])
            if isinstance(type_, int):
                key = rnd_int(values_len)
            else:
                indices = [random.choice([None, rnd_int(values_len)])
                           for _ in range(2)]
                indices.append(random.choice(
                    [None, rnd_nonzero_int(values_len)]))
                key = slice(*indices)
            keys.append(key)
        yield values, keys


getitem_n_delitem_cases = [
    ([], [0]),
    ([], [-1]),
    (['a'], [1]),
    (['a'], [-2]),
    (['a'], [0]),
    (['a'], [-1]),
    (['a', 'b'], [-2]),
    (['a', 'b'], [-1]),
    (['a', 'b', 'c'], [1, 0, 0]),
    ([], [slice(None, None, None)]),
    ([], [slice(1, 25, None)]),
    (['a'], [slice(None, 1, 0)]),
    (['a'], [slice(-13, -4, -13)]),
    (['a', 'b'], [slice(None, None, -1)]),
    (['a', 'b', 'c'], [slice(None, None, None)]),
    (['a', 'b', 'c'], [slice(None, None, 2)]),
    (['a', 'b', 'c', 'd', 'e'], [slice(None, None, None)]),
    (['a', 'b', 'c', 'd', 'e'], [slice(1, None, None)]),
    (['a', 'b', 'c', 'd', 'e'], [slice(1, None, 2)]),
    (['a', 'b', 'c', 'd', 'e'], [slice(None, -1, None)]),
    (['a', 'b', 'c', 'd', 'e'], [slice(None, None, -2)]),
    (['a', 'b', 'c', 'd', 'e'], [slice(None, None, -1)]),
    (['a', 'b', 'c', 'd', 'e'], [slice(2, -2, None)]),
    (['a', 'b', 'c', 'd', 'e'], [slice(-2, 2, -1)]),
    (['a', 'b', 'c', 'd', 'e'], [slice(6, 12, None)]),
    (['a', 'b', 'c', 'd', 'e'], [slice(6, 12, -1)]),
    (['a', 'b', 'c', 'd', 'e'], [slice(3, 55, None)]),
    (['a', 'b', 'c', 'd', 'e'], [slice(-1, -4, -1)]),
    (['a', 'b', 'c', 'd', 'e'], [slice(None, None, 3)]),
    (['a', 'b', 'c', 'd', 'e'], [slice(-4, None, 2)]),
    (['a', 'b', 'c', 'd', 'e'], [slice(None, 2, 2)]),
    (['a', 'b', 'c', 'd', 'e'], [slice(None, -3, 2)]),
    (['a', 'b', 'c', 'd', 'e'], [slice(2, None, -2)]),
    (['a', 'b', 'c', 'd', 'e'], [slice(-3, None, -2)]),
    (['a', 'b', 'c', 'd', 'e'], [slice(None, 2, -2)]),
    (['a', 'b', 'c', 'd', 'e'], [slice(None, -3, -2)]),
    (['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'], [slice(-2, 1, 11)]),
    (['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'], [slice(-6, 8, None)]),
    (
        ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
        [slice(None, None, 5)]
    ),
]


def test_getitem(values, keys):
    tree = SplayTree(values)
    assert check_whether_tree_is_correct(tree)
    expected_vals = list(values)
    for key in keys:
        expected_ex, expected_res = None, None
        try:
            expected_res = expected_vals[key]
        except Exception as e:  # pylint: disable=broad-except
            expected_ex = e
        ex, res = None, None
        try:
            res = tree[key]
            if isinstance(res, SplayTree):
                res = list(res)
        except Exception as e:  # pylint: disable=broad-except
            ex = e
        assert check_whether_tree_is_correct(tree)
        stmt = type(ex) is type(expected_ex) and res == expected_res
        assert stmt, f"case: {(values, keys)}"


def getitem_test_with_cases():
    padding = len(str(len(getitem_n_delitem_cases)))
    for i, case in enumerate(getitem_n_delitem_cases, 1):
        test_getitem(*case)
        print(f"getitem test for case {i:{padding}}"
              f"/{len(getitem_n_delitem_cases)} finished.", end='\r')
    print()


def getitem_random_test(
        how_many=1000, keys_maxlen=5, values_maxlen=10, values_source=None):
    padding = len(str(how_many))
    for i, case in enumerate(
            gen_cases_for_getitem_or_delitem(
                how_many, keys_maxlen, values_maxlen, values_source), 1):
        test_getitem(*case)
        print(f"getitem random test {i:{padding}}/{how_many} finished.",
              end='\r')
    print()


def test_delitem(values, keys):
    tree = SplayTree(values)
    assert check_whether_tree_is_correct(tree)
    expected_vals = list(values)
    for key in keys:
        expected_ex = None
        try:
            del expected_vals[key]
        except Exception as e:  # pylint: disable=broad-except
            expected_ex = e
        ex = None
        try:
            del tree[key]
        except Exception as e:  # pylint: disable=broad-except
            ex = e
        assert check_whether_tree_is_correct(tree)
        stmt = type(ex) is type(expected_ex) and list(tree) == expected_vals
        assert stmt, f"case: {(values, keys)}"


def delitem_test_with_cases():
    padding = len(str(len(getitem_n_delitem_cases)))
    for i, case in enumerate(getitem_n_delitem_cases, 1):
        test_delitem(*case)
        print(f"delitem test for case {i:{padding}}"
              f"/{len(getitem_n_delitem_cases)} finished.", end='\r')
    print()


def delitem_random_test(
        how_many=1000, keys_maxlen=5, values_maxlen=10, values_source=None):
    padding = len(str(how_many))
    for i, case in enumerate(
            gen_cases_for_getitem_or_delitem(
                how_many, keys_maxlen, values_maxlen, values_source), 1):
        test_delitem(*case)
        print(f"delitem random test {i:{padding}}/{how_many} finished.",
              end='\r')
    print()


def test_setitem(values, key_and_value_to_set_pairs):
    tree = SplayTree(values)
    assert check_whether_tree_is_correct(tree)
    expected_vals = list(values)
    for key, value in key_and_value_to_set_pairs:
        not_iterable_value = False
        try:
            val_it = iter(value)
        except TypeError:
            not_iterable_value = True
        if not_iterable_value or isinstance(key, int):
            list_value = value
            tree_value = value
        else:
            # val_it is bound because not_iterable_value is False
            # only if iter() call was successful
            list_value = list(val_it)
            tree_value = SplayTree(list_value)
            assert check_whether_tree_is_correct(tree_value)
        expected_ex = None
        try:
            expected_vals[key] = list_value
        except Exception as e:  # pylint: disable=broad-except
            expected_ex = e
        ex = None
        try:
            tree[key] = tree_value
        except Exception as e:  # pylint: disable=broad-except
            ex = e
        assert check_whether_tree_is_correct(tree)
        stmt = type(ex) is type(expected_ex) and list(tree) == expected_vals
        assert stmt, f"case: {(values, key_and_value_to_set_pairs)}"


def setitem_test_with_cases():
    class NotIterable:
        pass
    cases = [
        ([], [(0, 'b')]),
        (['a'], [(0, 'b')]),
        (['a', 'b', 'c'], [(1, 'd')]),
        (['a', 'b', 'c'], [(-1, 'd')]),
        ([], [(slice(None, None, None), [])]),
        ([], [(slice(None, None, None), NotIterable())]),
        ([], [(slice(None, None, 1), NotIterable())]),
        ([], [(slice(None, None, 2), NotIterable())]),
        ([], [(slice(None, None, -1), NotIterable())]),
        (['a'], [(slice(None, None, None), [])]),
        (['a'], [(slice(None, None, 1), [])]),
        (['a'], [(slice(None, None, 2), [])]),
        (['a'], [(slice(None, None, -1), [])]),
        ([], [(slice(None, None, None), ['a'])]),
        ([], [(slice(None, None, 1), ['a'])]),
        ([], [(slice(None, None, 2), ['a'])]),
        ([], [(slice(None, None, -1), ['a'])]),
        (['a', 'b', 'c'], [(slice(None, None, 1), ['d', 'e', 'f'])]),
        (['a', 'b', 'c'], [(slice(None, None, -1), ['d', 'e', 'f'])]),
        (['a', 'b', 'c'], [(slice(2, 1, 1), ['d', 'e', 'f'])]),
        (['a', 'b', 'c'], [(slice(2, 1, 1), [])]),
        (['a', 'b', 'c'], [(slice(-100, -100), ['d', 'e', 'f'])]),
        (['a', 'b', 'c'], [(slice(100, 100), ['d', 'e', 'f'])]),
        (['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
          'i', 'j'], [(slice(-2, 1, 11), ['k'])]),
        (['a', 'b', 'c', 'd', 'e'], [(slice(None, 2, 2), ['f'])]),
        (['a'], [(slice(-13, -4, -13), ['d'])]),
        (['a', 'b', 'c'], [(slice(None, None, 2), ['d'])]),
        (['a', 'b', 'c'], ((slice(-1, -3, None), ['d']),)),
    ]
    padding = len(str(len(cases)))
    for i, case in enumerate(cases, 1):
        test_setitem(*case)
        print(f"setitem test for case {i:{padding}}/{len(cases)} finished.",
              end='\r')
    print()


def gen_cases_for_setitem(
        how_many=1000, key_and_value_to_set_pairs_maxlen=5, values_maxlen=10,
        values_source=None):

    if values_source is None:
        values_source = string.ascii_lowercase
    values_source = sorted(set(values_source))
    if len(values_source) < values_maxlen:
        raise ValueError(
            '"values_source" must contain a number of unique values '
            'greater than or equal to "values_maxlen"')
    for _ in range(how_many):
        values_len = random.randint(0, values_maxlen)
        key_and_value_to_set_pairs_len = random.randint(
            1, key_and_value_to_set_pairs_maxlen)
        values = values_source[:values_maxlen]
        values_left = values_source[values_maxlen:]
        key_and_value_to_set_pairs = []
        for _ in range(key_and_value_to_set_pairs_len):
            type_ = random.choice([int, slice])
            if isinstance(type_, int):
                key = rnd_int(values_len)
                value_to_set = values_left.pop(0)
            else:
                indices = [random.choice([None, rnd_int(values_len)])
                           for _ in range(2)]
                indices.append(random.choice(
                    [None, rnd_nonzero_int(values_len)]))
                key = slice(*indices)
                value_to_set_len = random.randint(0, len(values_left))
                value_to_set = values_left[:value_to_set_len]
                del values_left[:value_to_set_len]
            key_and_value_to_set_pairs.append((key, value_to_set))
        yield values, key_and_value_to_set_pairs


def setitem_random_test(
        how_many=1000, keys_maxlen=5, values_maxlen=10, values_source=None):
    padding = len(str(how_many))
    for i, case in enumerate(
            gen_cases_for_setitem(
                how_many, keys_maxlen, values_maxlen, values_source), 1):
        test_setitem(*case)
        print(f"setitem random test {i:{padding}}/{how_many} finished.",
              end='\r')
    print()


if __name__ == "__main__":
    # main()
    test()
    # time_test()

    index_test_with_cases()
    index_test_with_special_cases()
    # index_random_test()

    append_test_with_cases()
    append_random_test()

    insert_test_with_special_cases()

    merge_test_with_cases()
    merge_random_test()

    split_test_with_cases()
    split_random_test()

    getitem_test_with_cases()
    getitem_random_test()

    delitem_test_with_cases()
    delitem_random_test()

    setitem_test_with_cases()
    setitem_random_test()
