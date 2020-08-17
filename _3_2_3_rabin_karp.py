from random import randint
from sys import stdin


def rabin_karp(pattern, text, prime=10 ** 9 + 7, *, do_not_checks=False):
    pattern_len, text_len = len(pattern), len(text)
    if not pattern_len or not text_len or pattern_len > text_len:
        return []
    if pattern_len >= prime:
        raise ValueError
    x = randint(1, prime - 1)

    def hash_(s):
        cur_x = 1
        last_char = -1
        chars_wo_last_in_reversed = slice(-2, None, -1)
        res = ord(s[last_char]) % prime
        for ch in s[chars_wo_last_in_reversed]:
            cur_x = cur_x * x % prime
            res += ord(ch) * cur_x % prime
        return res % prime, cur_x

    pattern_hash, _ = hash_(pattern)
    first_win_hash, x_p_1 = hash_(text[:pattern_len])

    maybe_occurrences = []
    if first_win_hash == pattern_hash:
        maybe_occurrences.append(0)

    prev_win_first_ch_i = 0
    cur_win_hash = first_win_hash
    for cur_win_last_ch_i in range(pattern_len, text_len):
        cur_win_hash -= ord(text[prev_win_first_ch_i]) * x_p_1 % prime
        cur_win_hash = cur_win_hash * x % prime
        cur_win_hash = (cur_win_hash + ord(text[cur_win_last_ch_i])) % prime
        if cur_win_hash == pattern_hash:
            maybe_occurrences.append(cur_win_last_ch_i - pattern_len + 1)
        prev_win_first_ch_i += 1

    if do_not_checks:
        return maybe_occurrences

    occurrences = []
    for win_start in maybe_occurrences:
        for pi, ti in enumerate(range(win_start, win_start + pattern_len)):
            if text[ti] != pattern[pi]:
                break
        else:
            occurrences.append(win_start)
    return occurrences


def main():
    pattern = stdin.readline().rstrip()
    text = stdin.readline().rstrip()
    print(*rabin_karp(pattern, text, do_not_checks=True))


def test():
    assert rabin_karp("aba", "abacaba") == [0, 4]
    assert rabin_karp("Test", "testTesttesT") == [4]
    assert rabin_karp("aaaaa", "baaaaaaa") == [1, 2, 3]
    assert rabin_karp("a", "baaab") == [1, 2, 3]
    assert rabin_karp("", "") == []
    assert rabin_karp("a", "") == []
    assert rabin_karp("", "a") == []


def time_test(number=1, pattern=None, text=None):
    import gc
    import random
    import string
    import timeit

    def prepare(text_len=5 * 10 ** 5):
        nonlocal pattern, text
        if pattern is None:
            if text is not None:
                text_len = len(text)
            pattern_len = random.randint(1, text_len)
            pattern = "".join(random.choices(
                string.ascii_letters, k=pattern_len))
        if text is None:
            text = "".join(random.choices(string.ascii_letters, k=text_len))
        return pattern, text

    print(timeit.timeit(stmt="gc.enable(); rabin_karp(*prepare())",
                        number=number, globals={**globals(), **locals()}) / number)


def time_test_one_symbol_pattern_case(number=1):
    time_test(number=number, pattern='a')


def time_test_worst_case(number=1):
    time_test(number=number, pattern="a" * 200, text="a" * 5 * 10 ** 5)
    # time_test(number=number, pattern="a" * 1, text="a" * 5 * 10 ** 5 * 200)


def total_occurrences_len(pattern="a" * 200, text="a" * 5 * 10 ** 5):
    print(len(rabin_karp(pattern, text)) * len(pattern))


if __name__ == "__main__":
    # main()
    # test()
    # time_test(5)
    # time_test_one_symbol_pattern_case(number=5)
    time_test_worst_case(1)
    # total_occurrences_len()
