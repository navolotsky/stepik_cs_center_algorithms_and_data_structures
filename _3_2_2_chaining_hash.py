from sys import stdin


class HashTable:
    def __init__(self, size, x=263, prime=10**9+7):
        self._size = size
        self._x = x
        self._prime = prime
        self._array = [[] for _ in range(size)]

    def _hash(self, s):
        x = 1
        res = ord(s[0]) % self._prime
        for ch in s[1:]:
            x = x * self._x % self._prime
            res += ord(ch) * x % self._prime
        return res % self._prime % self._size

    def add(self, s):
        hash_ = self._hash(s)
        for another_s in self._array[hash_]:
            if another_s == s:
                break
        else:
            self._array[hash_].append(s)

    def find(self, s):
        hash_ = self._hash(s)
        for another_s in self._array[hash_]:
            if another_s == s:
                return "yes"
        return "no"

    def delete(self, s):
        hash_ = self._hash(s)
        for i, another_s in enumerate(self._array[hash_]):
            if another_s == s:
                del self._array[hash_][i]
                break

    def check(self, i):
        return ' '.join(reversed(self._array[i]))


def make_queries(queries, table_size=10**7):
    table = HashTable(table_size)
    actions = {
        "add": table.add,
        "find": table.find,
        "del": table.delete,
        "check": table.check
    }
    result = []
    for action, *args in queries:
        res = actions[action](*args)
        if action in ("find", "check"):
            result.append(res)
    return result


def main():
    size = int(stdin.readline())
    _ = stdin.readline()
    queries = [
        [int(seq) if seq.isdigit() else seq
         for seq in line.split()] for line in stdin]
    print(*make_queries(queries, size), sep='\n')


def test():
    assert make_queries(
        [["add", "world"],
         ["add", "HellO"],
         ["check", 4],
         ["find", "World"],
         ["find", "world"],
         ["del", "world"],
         ["check", 4],
         ["del", "HellO"],
         ["add", "luck"],
         ["add", "GooD"],
         ["check", 2],
         ["del", "good"]],
        5
    ) == [
        "HellO world",
        "no",
        "yes",
        "HellO",
        "GooD luck"
    ]
    assert make_queries(
        [["add", "test"],
         ["add", "test"],
         ["find", "test"],
         ["del", "test"],
         ["find", "test"],
         ["find", "Test"],
         ["add", "Test"],
         ["find", "Test"]],
        4
    ) == [
        "yes",
        "no",
        "no",
        "yes"
    ]
    assert make_queries(
        [["check", 0],
         ["find", "help"],
         ["add", "help"],
         ["add", "del"],
         ["add", "add"],
         ["find", "add"],
         ["find", "del"],
         ["del", "del"],
         ["find", "del"],
         ["check", 0],
         ["check", 1],
         ["check", 2]],
        3
    ) == [
        "",
        "no",
        "yes",
        "yes",
        "no",
        "",
        "add help",
        ""
    ]


def time_test(number=10):
    import random
    import gc
    import timeit
    import string
    import math

    def prepare(how_many=10**5):
        size = random.randint(math.ceil(how_many / 5), how_many)
        queries = []
        for _ in range(how_many):
            action = random.choice(["add", "find", "del", "check"])
            args = [action]
            if action == "check":
                args.append(random.randint(0, size - 1))
            else:
                args.append("".join(random.choices(
                    string.ascii_letters, k=15)))
            queries.append(args)
        return queries, size
    print(timeit.timeit(stmt="gc.enable(); make_queries(*prepare())",
                        number=number, globals={**globals(), **locals()})/number)


if __name__ == "__main__":
    main()
    # test()
    # time_test()
