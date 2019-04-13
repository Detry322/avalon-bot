#include "lookup_tables.h"

const int PROPOSAL_TO_INDEX_LOOKUP[32] = {-1, -1, -1, 0, -1, 1, 4, 0, -1, 2, 5, 1, 7, 3, 6, -1, -1, 3, 6, 2, 8, 4, 7, -1, 9, 5, 8, -1, 9, -1, -1, -1};
const int INDEX_TO_PROPOSAL_2[10] = {3, 5, 9, 17, 6, 10, 18, 12, 20, 24};
const int INDEX_TO_PROPOSAL_3[10] = {7, 11, 19, 13, 21, 25, 14, 22, 26, 28};
const int ROUND_TO_PROPOSE_SIZE[5] = {2, 3, 2, 3, 3};


const int VIEWPOINT_TO_BAD[NUM_PLAYERS][NUM_VIEWPOINTS] = {
    { -1,    6,   10,   18,   12,   20,   24,    1,    2,    3,    4,    1,    2,    3,    4 },
    { -1,    5,    9,   17,   12,   20,   24,    0,    2,    3,    4,    0,    2,    3,    4 },
    { -1,    3,    9,   17,   10,   18,   24,    0,    1,    3,    4,    0,    1,    3,    4 },
    { -1,    3,    5,   17,    6,   18,   20,    0,    1,    2,    4,    0,    1,    2,    4 },
    { -1,    3,    5,    9,    6,   10,   12,    0,    1,    2,    3,    0,    1,    2,    3 }
};


const int ASSIGNMENT_TO_VIEWPOINT[NUM_ASSIGNMENTS][NUM_PLAYERS] = {
    {  1,    8,   12,    0,    0 },
    {  2,    9,    0,   12,    0 },
    {  3,   10,    0,    0,   12 },
    {  1,   12,    8,    0,    0 },
    {  4,    0,    9,   13,    0 },
    {  5,    0,   10,    0,   13 },
    {  2,   13,    0,    8,    0 },
    {  4,    0,   13,    9,    0 },
    {  6,    0,    0,   10,   14 },
    {  3,   14,    0,    0,    8 },
    {  5,    0,   14,    0,    9 },
    {  6,    0,    0,   14,   10 },
    {  8,    1,   11,    0,    0 },
    {  9,    2,    0,   11,    0 },
    { 10,    3,    0,    0,   11 },
    { 12,    1,    7,    0,    0 },
    {  0,    4,    9,   13,    0 },
    {  0,    5,   10,    0,   13 },
    { 13,    2,    0,    7,    0 },
    {  0,    4,   13,    9,    0 },
    {  0,    6,    0,   10,   14 },
    { 14,    3,    0,    0,    7 },
    {  0,    5,   14,    0,    9 },
    {  0,    6,    0,   14,   10 },
    {  7,   11,    1,    0,    0 },
    {  9,    0,    2,   11,    0 },
    { 10,    0,    3,    0,   11 },
    { 11,    7,    1,    0,    0 },
    {  0,    9,    4,   12,    0 },
    {  0,   10,    5,    0,   12 },
    { 13,    0,    2,    7,    0 },
    {  0,   13,    4,    8,    0 },
    {  0,    0,    6,   10,   14 },
    { 14,    0,    3,    0,    7 },
    {  0,   14,    5,    0,    8 },
    {  0,    0,    6,   14,   10 },
    {  7,   11,    0,    1,    0 },
    {  8,    0,   11,    2,    0 },
    { 10,    0,    0,    3,   11 },
    { 11,    7,    0,    1,    0 },
    {  0,    8,   12,    4,    0 },
    {  0,   10,    0,    5,   12 },
    { 12,    0,    7,    2,    0 },
    {  0,   12,    8,    4,    0 },
    {  0,    0,   10,    6,   13 },
    { 14,    0,    0,    3,    7 },
    {  0,   14,    0,    5,    8 },
    {  0,    0,   14,    6,    9 },
    {  7,   11,    0,    0,    1 },
    {  8,    0,   11,    0,    2 },
    {  9,    0,    0,   11,    3 },
    { 11,    7,    0,    0,    1 },
    {  0,    8,   12,    0,    4 },
    {  0,    9,    0,   12,    5 },
    { 12,    0,    7,    0,    2 },
    {  0,   12,    8,    0,    4 },
    {  0,    0,    9,   13,    6 },
    { 13,    0,    0,    7,    3 },
    {  0,   13,    0,    8,    5 },
    {  0,    0,   13,    9,    6 }
};


const int ASSIGNMENT_TO_EVIL[NUM_ASSIGNMENTS] = {  6,  10,  18,   6,  12,  20,  10,  12,  24,  18,  20,  24,   5,   9,  17,   5,  12,  20,   9,  12,  24,  17,  20,  24,   3,   9,  17,   3,  10,  18,   9,  10,  24,  17,  18,  24,   3,   5,  17,   3,   6,  18,   5,   6,  20,  17,  18,  20,   3,   5,   9,   3,   6,  10,   5,   6,  12,   9,  10,  12 };


const int VIEWPOINT_TO_PARTNER_VIEWPOINT[NUM_PLAYERS][NUM_VIEWPOINTS] = {
    { -1,   -1,   -1,   -1,   -1,   -1,   -1,   11,   11,   11,   11,    7,    7,    7,    7 },
    { -1,   -1,   -1,   -1,   -1,   -1,   -1,   11,   12,   12,   12,    7,    8,    8,    8 },
    { -1,   -1,   -1,   -1,   -1,   -1,   -1,   12,   12,   13,   13,    8,    8,    9,    9 },
    { -1,   -1,   -1,   -1,   -1,   -1,   -1,   13,   13,   13,   14,    9,    9,    9,   10 },
    { -1,   -1,   -1,   -1,   -1,   -1,   -1,   14,   14,   14,   14,   10,   10,   10,   10 }
};

