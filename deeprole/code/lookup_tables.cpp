#include "lookup_tables.h"

const int PROPOSAL_TO_INDEX_LOOKUP[32] = {-1, -1, -1, 0, -1, 1, 4, 0, -1, 2, 5, 1, 7, 3, 6, -1, -1, 3, 6, 2, 8, 4, 7, -1, 9, 5, 8, -1, 9, -1, -1, -1};
const int INDEX_TO_PROPOSAL_2[10] = {3, 5, 9, 17, 6, 10, 18, 12, 20, 24};
const int INDEX_TO_PROPOSAL_3[10] = {7, 11, 19, 13, 21, 25, 14, 22, 26, 28};
const int ROUND_TO_PROPOSE_SIZE[5] = {2, 3, 2, 3, 3};
