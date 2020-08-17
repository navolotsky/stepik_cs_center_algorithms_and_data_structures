from sys import stdin


class PhoneBook:
    def __init__(self, record_num):
        self._array = [None] * record_num

    def add(self, number, name):
        self._array[number] = name

    def find(self, number):
        name = self._array[number]
        return name if name is not None else "not found"

    def delete(self, number):
        self._array[number] = None


def make_queries(queries, record_number=10**7):
    phone_book = PhoneBook(record_number)
    actions = {
        "add": phone_book.add,
        "find": phone_book.find,
        "del": phone_book.delete
    }
    result = []
    for action, *args in queries:
        res = actions[action](*args)
        if action == "find":
            result.append(res)
    return result


def main():
    _ = int(stdin.readline())
    queries = [
        [int(seq) if seq.isdigit() else seq
         for seq in line.split()] for line in stdin]
    print(*make_queries(queries), sep='\n')


def test():
    assert make_queries([
        ["add", 911, "police"],
        ["add", 76213, "Mom"],
        ["add", 17239, "Bob"],
        ["find", 76213],
        ["find", 910],
        ["find", 911],
        ["del", 910],
        ["del", 911],
        ["find", 911],
        ["find", 76213],
        ["add", 76213, "daddy"],
        ["find", 76213]]
    ) == [
        "Mom",
        "not found",
        "police",
        "not found",
        "Mom",
        "daddy"
    ]
    assert make_queries([
        ['find', 3839442],
        ['add', 123456, 'me'],
        ['add', 0, 'granny'],
        ['find', 0],
        ['find', 123456],
        ['del', 0],
        ['del', 0],
        ['find', 0]]
    ) == [
        "not found",
        "granny",
        "me",
        "not found"
    ]


def time_test(number=10):
    import random
    import gc
    import timeit
    import string

    def prepare(how_many=10**5):
        queries = []
        for _ in range(how_many):
            action = random.choice(["add", "find", "del"])
            args = [action]
            args.append(random.randint(1, 10**7))
            if action == "add":
                args.append("".join(random.choices(
                    string.ascii_letters, k=15)))
            queries.append(args)
        return queries
    print(timeit.timeit(stmt="gc.enable(); make_queries(prepare())",
                        number=number, globals={**globals(), **locals()})/number)


if __name__ == "__main__":
    main()
    # test()
    # time_test()
