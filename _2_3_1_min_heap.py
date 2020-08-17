import sys


class MinHeap:
    def __init__(self, array=None):
        self._swaps = []
        if array is None:
            self._array = []
        else:
            self._array = array
            self._build()

    def __bool__(self):
        return bool(self._array)

    def _swap(self, i, j):
        self._array[i], self._array[j] = self._array[j], self._array[i]
        self._swaps.append((i, j))

    def _sift_up(self, ind):
        while ind != 0:
            parent = (ind - 1) // 2
            if self._array[ind] < self._array[parent]:
                self._swap(parent, ind)
            else:
                break
            ind = parent

    def _sift_down(self, ind):
        min_ind = ind
        arr_len = len(self._array)
        while True:
            left = 2 * ind + 1
            right = left + 1
            if left < arr_len and self._array[left] < self._array[min_ind]:
                min_ind = left
            if right < arr_len and self._array[right] < self._array[min_ind]:
                min_ind = right
            if ind != min_ind:
                self._swap(ind, min_ind)
                ind = min_ind
            else:
                break

    def extract_min(self):
        self._swaps = []
        self._swap(0, -1 + len(self._array))
        result = self._array.pop()
        self._sift_down(0)
        return result

    def insert(self, element):
        self._swaps = []
        self._array.append(element)
        self._sift_up(-1 + len(self._array))

    def _build(self):
        for i in range((-1 + len(self._array)) // 2, -1, -1):
            self._sift_down(i)

    @classmethod
    def heapify(cls, array):
        instance = cls(array)
        instance._build()
        return len(instance._swaps), instance._swaps


def main():
    _ = sys.stdin.readline()
    array = list(map(int, sys.stdin.readline().split()))
    count, swaps = MinHeap.heapify(array)
    print(count)
    print(*["{} {}".format(*swap) for swap in swaps], sep='\n')


def test():
    assert MinHeap.heapify([5, 4, 3, 2, 1]) == (3, [
        (1, 4),
        (0, 1),
        (1, 3)])
    assert MinHeap.heapify([1, 2, 3, 4, 5]) == (0, [])
    assert MinHeap.heapify([0, 1, 2, 3, 4, 5]) == (0, [])
    assert MinHeap.heapify([7, 6, 5, 4, 3, 2]) == (4, [
        (2, 5),
        (1, 4),
        (0, 2),
        (2, 5)])


def time_test(number=1):
    import random
    import gc
    import timeit
    print(timeit.timeit(
        stmt="MinHeap.heapify(array)",
        setup="gc.enable(); array=[random.randint(0, 10**9) for _ in range(10**5)]",
        number=number,
        globals={**locals(), **globals()})/number)


if __name__ == "__main__":
    main()
    # test()
    # time_test(10)
