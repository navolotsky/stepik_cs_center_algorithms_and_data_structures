from sys import stdin


def make_tables(num):
    return list(range(num)), [0] * num


def find(parent, i):
    if i != parent[i]:
        parent[i] = find(parent, parent[i])
    return parent[i]


def union(parent, rank, size, dst, src):
    new_size = -1
    i_id, j_id = find(parent, dst), find(parent, src)
    if i_id == j_id:
        return new_size
    if rank[i_id] > rank[j_id]:
        parent[j_id] = i_id
        size[i_id] += size[j_id]
        size[j_id] = 0
        new_size = size[i_id]
    else:
        parent[i_id] = j_id
        size[j_id] += size[i_id]
        size[i_id] = 0
        if rank[i_id] == rank[j_id]:
            rank[j_id] += 1
        new_size = size[j_id]
    return new_size


def max_sizes_after_joins(parent, rank, size, ops):
    result = []
    max_size = max(size)
    for dst, src in ops:
        res = union(parent, rank, size, dst - 1, src - 1)
        max_size = max(max_size, res)
        result.append(max_size)
    return result


def main():
    num, _ = map(int, stdin.readline().split())
    size = list(map(int, stdin.readline().split()))
    ops = [tuple(map(int, line.split())) for line in stdin]

    parent, rank = make_tables(num)
    max_sizes = max_sizes_after_joins(parent, rank, size, ops)
    print(*max_sizes, sep='\n')


def test():
    assert max_sizes_after_joins(
        *make_tables(5),
        [1] * 5,
        [(3, 5),
         (2, 4),
         (1, 4),
         (5, 4),
         (5, 3)]) == [2, 2, 3, 5, 5]
    assert max_sizes_after_joins(
        *make_tables(6),
        [10, 0, 5, 0, 3, 3],
        [(6, 6),
         (6, 5),
         (5, 4),
         (4, 3)
         ]) == [10, 10, 10, 11]
    assert max_sizes_after_joins(
        *make_tables(1),
        [0],
        [(0, 0), (0, 0)]
    ) == [0, 0]
    assert max_sizes_after_joins(
        *make_tables(1),
        [5],
        [(0, 0), (0, 0)]
    ) == [5, 5]


def time_test(number=5):
    import random
    import gc
    import timeit
    tables_num = ops_num = 10**5
    print(timeit.timeit(
        stmt=(
            "gc.enable();"
            "max_sizes_after_joins("
                "*make_tables(tables_num),"
                "[random.randint(0, 10**4) for _ in range(tables_num)],"
                "[(random.randint(0, tables_num), random.randint(0, tables_num))"
                "for _ in range(ops_num)])"),
        number=number,
        globals={**locals(), **globals()})/number)


if __name__ == "__main__":
    main()
    # test()
    # time_test()
