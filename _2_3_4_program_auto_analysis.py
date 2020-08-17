from sys import stdin


def make_sets(num):
    return list(range(num)), [0] * num


def find(parent, i):
    if i != parent[i]:
        parent[i] = find(parent, parent[i])
    return parent[i]


def union(parent, rank, i, j):
    i_id = find(parent, i)
    j_id = find(parent, j)
    if i_id == j_id:
        return
    if rank[i_id] > rank[j_id]:
        parent[j_id] = i_id
    else:
        parent[i_id] = j_id
        if rank[i_id] == rank[j_id]:
            rank[j_id] += 1


def analysis(variables_number, equalities, inequalities):
    parent, rank = make_sets(variables_number)
    for x, y in equalities:
        union(parent, rank, x - 1, y - 1)
    for x, y in inequalities:
        if find(parent, x - 1) == find(parent, y - 1):
            return 0
    return 1


def main():
    var_num, eq_num, ineq_num = map(int, stdin.readline().split())
    eqs = [tuple(map(int, stdin.readline().split())) for _ in range(eq_num)]
    ineqs = [tuple(map(int, stdin.readline().split()))
             for _ in range(ineq_num)]
    print(analysis(var_num, eqs, ineqs))


def test():
    assert analysis(4,
                    [
                        (1, 2),
                        (1, 3),
                        (1, 4),
                        (2, 3),
                        (2, 4),
                        (3, 4)
                    ],
                    []
                    ) == 1
    assert analysis(6,
                    [
                        (2, 3),
                        (1, 5),
                        (2, 5),
                        (3, 4),
                        (4, 2)],
                    [
                        (6, 1),
                        (4, 6),
                        (4, 5)]
                    ) == 0
    assert analysis(1, [], []) == 1
    assert analysis(1, [], [(1, 1)]) == 0
    assert analysis(1, [(1, 1)], []) == 1


def time_test(number=1):
    import random
    import timeit
    import gc
    var_num = 10**5
    eq_plus_ineq_num = 2*10**5

    def prepare():
        while True:
            eq_num = random.randint(0, eq_plus_ineq_num)
            ineq_num = random.randint(0, eq_plus_ineq_num)
            if eq_num + ineq_num <= eq_plus_ineq_num:
                break
        eqs = [(random.randint(1, var_num), random.randint(1, var_num))
               for _ in range(eq_num)]
        ineqs = [(random.randint(1, var_num), random.randint(1, var_num))
                 for _ in range(ineq_num)]
        return eqs, ineqs
    print(timeit.timeit(stmt="analysis(var_num, *prepare())",
                        number=number, globals={**locals(), **globals()})/number)


if __name__ == "__main__":
    main()
    # test()
    # time_test(5)
