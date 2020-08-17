import sys


class MinHeap:
    def __init__(self, array=None):
        if array is None:
            self._array = []
        else:
            self._array = array
            self._build()

    def __bool__(self):
        return bool(self._array)

    def _swap(self, i, j):
        self._array[i], self._array[j] = self._array[j], self._array[i]

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

    def _build(self):
        for i in range((-1 + len(self._array)) // 2, -1, -1):
            self._sift_down(i)

    def get_min(self):
        if not self._array:
            return None
        return self._array[0]

    def extract_min(self):
        self._swap(0, -1 + len(self._array))
        result = self._array.pop()
        self._sift_down(0)
        return result

    def insert(self, element):
        self._array.append(element)
        self._sift_up(-1 + len(self._array))

    @classmethod
    def heapify(cls, array):
        cls(array)


class Processor:
    def __init__(self, number):
        self._number = number
        self._task = None
        self._start_time = 0
        self._end_time = 0

    def __lt__(self, other):
        return (self._end_time < other._end_time) or (
            self._end_time == other._end_time and self._number < other._number)

    def process_task(self, task_index, task_time):
        self._task = task_index
        self._start_time = self._end_time
        self._end_time += task_time

    def last_task_info(self):
        return self._task, self._number, self._start_time


def simulate(proc_num, tasks):
    def get_result(processor):
        nonlocal result
        ind, *info = processor.last_task_info()
        if ind is not None:
            result[ind] = info
    heap = MinHeap()
    proc_count = 0
    result = [None] * len(tasks)
    for ind, time in enumerate(tasks):
        if proc_count < proc_num and (not heap or heap.get_min()._end_time != 0):
            processor = Processor(proc_count)
            proc_count += 1
        else:
            processor = heap.extract_min()
            get_result(processor)
        processor.process_task(ind, time)
        heap.insert(processor)
    while heap:
        get_result(heap.extract_min())
    return result


def main():
    proc_num, _ = map(int, sys.stdin.readline().split())
    tasks = list(map(int, sys.stdin.readline().split()))
    print(*["{} {}".format(*info)
            for info in simulate(proc_num, tasks)], sep='\n')


def test():
    assert simulate(2, [1, 2, 3, 4, 5]) == [
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 2],
        [0, 4]
    ]
    assert simulate(4, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]) == [
        [0, 0],
        [1, 0],
        [2, 0],
        [3, 0],
        [0, 1],
        [1, 1],
        [2, 1],
        [3, 1],
        [0, 2],
        [1, 2],
        [2, 2],
        [3, 2],
        [0, 3],
        [1, 3],
        [2, 3],
        [3, 3],
        [0, 4],
        [1, 4],
        [2, 4],
        [3, 4]
    ]
    assert simulate(98, [0, 0]) == [
        [0, 0],
        [0, 0]
    ]
    assert simulate(98, [1, 10]) == [
        [0, 0],
        [1, 0]
    ]


def time_test(number=1):
    import random
    import gc
    import timeit
    # proc_num = random.randint(1, 10**5)
    # proc_num = 1000
    print(timeit.timeit(
        stmt="simulate(proc_num, tasks)",
        setup="gc.enable(); tasks=[random.randint(0, 10**9) for _ in range(10**5)]; proc_num = 10**5",
        number=number,
        globals={**locals(), **globals()})/number)


if __name__ == "__main__":
    # main()
    test()
    # time_test(1)
