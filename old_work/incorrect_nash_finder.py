##### this assumes everyone knows who the bad guys are.

from collections import namedtuple

GameState = namedtuple('GameState', ['bad', 'proposal1', 'proposal1_succeed', 'proposal2', 'proposal2_succeed', 'proposal3', 'proposal3_succeed'])

GAMES = []

BAD = 0
PROPOSAL_1 = 1
PROPOSAL_1_SUCCEED = 2
PROPOSAL_2 = 3
PROPOSAL_2_SUCCEED = 4
PROPOSAL_3 = 5
PROPOSAL_3_SUCCEED = 6


for bad in range(3):
  for proposal1 in [(0,1), (0, 2), (1,2)]:
    for succeed1 in [True, False]:
      for proposal2 in [(0,1), (0, 2), (1,2)]:
        for succeed2 in [True, False]:
          for proposal3 in [(0,1), (0, 2), (1,2)]:
            for succeed3 in [True, False]:
              if not succeed1 and bad not in proposal1:
                continue
              if not succeed2 and bad not in proposal2:
                continue
              if not succeed3 and bad not in proposal3:
                continue
              GAMES.append(GameState(bad, proposal1, succeed1, proposal2, succeed2, proposal3, succeed3))

# State to if the good people win, next_state
ACTION_TABLE = {}

for game in GAMES:
    ACTION_TABLE[game] = ((int(game.proposal1_succeed) + int(game.proposal2_succeed) + int(game.proposal3_succeed)) >= 2, None)

####
#  Proposal 3    Fail/Succeed
####
for game in GAMES:
    subgame = game[:PROPOSAL_3_SUCCEED]
    if subgame[BAD] not in subgame[PROPOSAL_3]:
        next_state = subgame + (True, )
        result, _ = ACTION_TABLE[next_state]
        ACTION_TABLE[subgame] = (result, next_state)
    else:
        possibility = {}
        for proposal_3_succeed in [True, False]:
            next_state = subgame + (proposal_3_succeed,)
            result, _ = ACTION_TABLE[next_state]
            possibility[result] = next_state
        if False in possibility:
            ACTION_TABLE[subgame] = (False, possibility[False])
        else:
            ACTION_TABLE[subgame] = (True, possibility[True])

####
#  Proposal 3
####

for game in GAMES:
    subgame = game[:PROPOSAL_3]
    possibility = {}
    for proposal in [(0, 1), (0, 2), (1, 2)]:
        next_state = subgame + (proposal,)
        result, _ = ACTION_TABLE[next_state]
        possibility[result] = next_state
    if subgame[BAD] == 2 and False in possibility:
        ACTION_TABLE[subgame] = (False, possibility[False])
    elif True in possibility:
        ACTION_TABLE[subgame] = (True, possibility[True])
    else:
        ACTION_TABLE[subgame] = (False, possibility[False])


####
#  Proposal 2    Fail/Succeed
####

for game in GAMES:
    subgame = game[:PROPOSAL_2_SUCCEED]
    if subgame[BAD] not in subgame[PROPOSAL_2]:
        next_state = subgame + (True, )
        result, _ = ACTION_TABLE[next_state]
        ACTION_TABLE[subgame] = (result, next_state)
    else:
        possibility = {}
        for proposal_2_succeed in [True, False]:
            next_state = subgame + (proposal_2_succeed,)
            result, _ = ACTION_TABLE[next_state]
            possibility[result] = next_state
        if False in possibility:
            ACTION_TABLE[subgame] = (False, possibility[False])
        else:
            ACTION_TABLE[subgame] = (True, possibility[True])

####
#  Proposal 2
####


for game in GAMES:
    subgame = game[:PROPOSAL_2]
    possibility = {}
    for proposal in [(0, 1), (0, 2), (1, 2)]:
        next_state = subgame + (proposal,)
        result, _ = ACTION_TABLE[next_state]
        possibility[result] = next_state
    if subgame[BAD] == 1 and False in possibility:
        ACTION_TABLE[subgame] = (False, possibility[False])
    elif True in possibility:
        ACTION_TABLE[subgame] = (True, possibility[True])
    else:
        ACTION_TABLE[subgame] = (False, possibility[False])

####
#  Proposal 1    Fail/Succeed
####


for game in GAMES:
    subgame = game[:PROPOSAL_1_SUCCEED]
    if subgame[BAD] not in subgame[PROPOSAL_1]:
        next_state = subgame + (True, )
        result, _ = ACTION_TABLE[next_state]
        ACTION_TABLE[subgame] = (result, next_state)
    else:
        possibility = {}
        for proposal_1_succeed in [True, False]:
            next_state = subgame + (proposal_1_succeed,)
            result, _ = ACTION_TABLE[next_state]
            possibility[result] = next_state
        if False in possibility:
            ACTION_TABLE[subgame] = (False, possibility[False])
        else:
            ACTION_TABLE[subgame] = (True, possibility[True])


####
#  Proposal 1
####

for game in GAMES:
    subgame = game[:PROPOSAL_1]
    possibility = {}
    for proposal in [(0, 1), (0, 2), (1, 2)]:
        next_state = subgame + (proposal,)
        result, _ = ACTION_TABLE[next_state]
        possibility[result] = next_state
    if subgame[BAD] == 0 and False in possibility:
        ACTION_TABLE[subgame] = (False, possibility[False])
    elif True in possibility:
        ACTION_TABLE[subgame] = (True, possibility[True])
    else:
        ACTION_TABLE[subgame] = (False, possibility[False])



ROUNDS = ["Proposal 1", "Proposal 1 succeed", "Proposal 2", "Proposal 2 succeed", "Proposal 3", "Proposal 3 succeed", "Endgame"]
#####

state = (0, )
print("======= Nash for bad guy 0 =======")
print("Good guys win: {}".format(ACTION_TABLE[state][0]))
print("====== Game play ======")

r = 0
while state is not None:
    _, state = ACTION_TABLE[state]
    if state is None:
        continue
    print("{}: {}".format(ROUNDS[r], state[-1]))
    r += 1


state = (1, )
print("======= Nash for bad guy 1 =======")
print("Good guys win: {}".format(ACTION_TABLE[state][0]))
print("====== Game play ======")

r = 0
while state is not None:
    _, state = ACTION_TABLE[state]
    if state is None:
        continue
    print("{}: {}".format(ROUNDS[r], state[-1]))
    r += 1


state = (2, )
print("======= Nash for bad guy 2 =======")
print("Good guys win: {}".format(ACTION_TABLE[state][0]))
print("====== Game play ======")

r = 0
while state is not None:
    _, state = ACTION_TABLE[state]
    if state is None:
        continue
    print("{}: {}".format(ROUNDS[r], state[-1]))
    r += 1
