#ifndef LOOKUP_TABLES_H_
#define LOOKUP_TABLES_H_

#define NUM_PLAYERS 5
#define NUM_EVIL 2
#define NUM_VIEWPOINTS (1 + 6 + 4 + 4)
#define NUM_GOOD_VIEWPOINTS (1 + 6)
#define NUM_ASSIGNMENTS 60
#define NUM_PROPOSAL_OPTIONS 10
#define TREMBLE_VALUE 1e-25

#define EVIL_WIN_PAYOFF 1.5
#define EVIL_LOSE_PAYOFF -1.5
#define GOOD_WIN_PAYOFF 1.0
#define GOOD_LOSE_PAYOFF -1.0

extern const int PROPOSAL_TO_INDEX_LOOKUP[32];
extern const int INDEX_TO_PROPOSAL_2[10];
extern const int INDEX_TO_PROPOSAL_3[10];
extern const int ROUND_TO_PROPOSE_SIZE[5];
extern const int VIEWPOINT_TO_BAD[NUM_PLAYERS][NUM_VIEWPOINTS];
extern const int ASSIGNMENT_TO_VIEWPOINT[NUM_ASSIGNMENTS][NUM_PLAYERS];
extern const int ASSIGNMENT_TO_EVIL[NUM_ASSIGNMENTS];
extern const int ASSIGNMENT_TO_ROLES[NUM_ASSIGNMENTS][3]; // merlin, assassin, minion

// Only applicable to 5 and 6 player avalon.
extern const int VIEWPOINT_TO_PARTNER_VIEWPOINT[NUM_PLAYERS][NUM_VIEWPOINTS];

#endif // LOOKUP_TABLES_H_
