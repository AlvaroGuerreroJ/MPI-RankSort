#!/usr/bin/env bash

for n_elems in {100000..1000000..100000}; do
    echo
    printf "Testing %s elements\n" "$n_elems"
    for i in {1..15}; do
        # mpirun -np 9 ./sort -r -t -n "$n_elems"
        # ./qsort.out "$n_elems"
        mpirun -np 9 rank_sort.out "$n_elems"
        echo
    done |&
        tee >(python3 get-avg.py "Total: ([0-9\.]+)")

    echo
done
