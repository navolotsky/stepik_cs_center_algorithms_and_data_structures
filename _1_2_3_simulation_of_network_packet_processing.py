import sys
import bisect


def simulate(buf_size, packets):
    queue = list(reversed(packets))
    processing_ends = []
    result = []
    total_time = 0
    while queue:
        arrival, duration = queue.pop()
        del processing_ends[:bisect.bisect(processing_ends, arrival)]
        if len(processing_ends) >= buf_size:
            result.append(-1)
            continue
        if arrival > total_time:
            total_time = arrival
        result.append(total_time)
        total_time += duration
        processing_ends.append(total_time)
    return result


def main():
    size, num = map(int, sys.stdin.readline().split())
    packets = []
    for _ in range(num):
        packets.append(tuple(map(int, sys.stdin.readline().split())))
    result = simulate(size, packets)
    if result:
        print(*result, sep='\n')


def test():
    assert simulate(1, []) == []
    assert simulate(1, [(0, 0)]) == [0]
    assert simulate(1, [(0, 1)]) == [0]
    assert simulate(1, [(0, 1), (0, 1)]) == [0, -1]
    assert simulate(1, [(0, 1), (1, 1)]) == [0, 1]
    assert simulate(1, [(0, 1), (1, 1), (1, 1)]) == [0, 1, -1]
    assert simulate(1, [(0, 1), (1, 3), (1, 1), (4, 1), (4, 1)]) == [
        0, 1, -1, 4, -1]
    assert simulate(1, [(0, 5), (0, 1), (5, 7), (6, 8), (12, 3)]) == [0, -1, 5, -1, 12]
    assert simulate(1, [(0, 1), (5, 10)]) == [0, 5]


def time_test(number=1):
    import timeit
    import random
    import string
    import gc
    size = 10**5
    n = 10**5
    print(timeit.timeit(
        stmt="simulate(size, packets)",
        setup="gc.enable(); packets = [(random.randint(0, 10**6), random.randint(0, 10**3))]\nfor i in range(1, n): packets.append((random.randint(packets[i-1][0], 10**6), random.randint(0, 10**3)))",
        globals={**globals(), **locals()}, number=number)/number)


if __name__ == "__main__":
    main()
    # test()
    # time_test(5)
