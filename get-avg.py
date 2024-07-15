#!/usr/bin/env python3
import sys
import re

def main():
    regex_to_math = sys.argv[1]

    count = 0
    sum_ = 0

    for line in sys.stdin:
        m = re.match(regex_to_math, line)

        if m is None:
            continue

        n = float(m.group(1))
        sum_ += n
        count += 1

    print(sum_, count)
    print(sum_ / count)


if __name__ == '__main__':
    main()
