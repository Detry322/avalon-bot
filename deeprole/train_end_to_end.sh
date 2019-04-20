#!/usr/bin/env bash

function train_section() {
    NUM_SUCCEEDS="$1"
    NUM_FAILS="$2"
    PROPOSE_COUNT="$3"

    echo $NUM_SUCCEEDS, $NUM_FAILS, $PROPOSE_COUNT
}

ITEMS="2 2
2 1
1 2
2 0
1 1
0 2
1 0
0 1
0 0"

IFS=$'\n'
for item in $ITEMS; do
    for i in $(seq 4 0); do
        IFS=' ' read ns nf <<< "$item"
        train_section $ns $nf $i
    done
done
