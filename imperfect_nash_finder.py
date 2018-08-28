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
    ACTION_TABLE[game] = (float((int(game.proposal1_succeed) + int(game.proposal2_succeed) + int(game.proposal3_succeed)) >= 2), [])


# Only works since there's three players btw.
def player_knows_bad_guy(player, state):
    if state[BAD] == player:
        return True
    if len(state) > PROPOSAL_1_SUCCEED:
        if player in state[PROPOSAL_1] and state[BAD] in state[PROPOSAL_1] and not state[PROPOSAL_1_SUCCEED]:
            return True
    if len(state) > PROPOSAL_2_SUCCEED:
        if player in state[PROPOSAL_2] and state[BAD] in state[PROPOSAL_2] and not state[PROPOSAL_2_SUCCEED]:
            return True
    if len(state) > PROPOSAL_3_SUCCEED:
        if player in state[PROPOSAL_3] and state[BAD] in state[PROPOSAL_3] and not state[PROPOSAL_3_SUCCEED]:
            return True

    return False


####
#  Proposal 3    Fail/Succeed
####
for game in GAMES:
    subgame = game[:PROPOSAL_3_SUCCEED]
    if subgame[BAD] not in subgame[PROPOSAL_3]:
        next_state = subgame + (True, )
        result, _ = ACTION_TABLE[next_state]
        ACTION_TABLE[subgame] = (result, [next_state])
        # good guys must succeed a vote. 
    else:
        # bad guy does the thing that minimizes good guy's success chance.
        best_option = min([False, True], key=lambda vote: ACTION_TABLE[subgame + (vote,)][0])
        next_state = subgame + (best_option, )
        result, _ = ACTION_TABLE[next_state]
        ACTION_TABLE[subgame] = (result, [next_state])

####
#  Proposal 3
####

for game in GAMES:
    subgame = game[:PROPOSAL_3]
    if subgame[BAD] == 2:
        # player 3 is the bad guy
        # Play the proposal that minimizes the good guy's chance of success.
        best_proposal = min([(1, 2), (0, 2), (0, 1)], key=lambda proposal: ACTION_TABLE[subgame + (proposal,)][0])
        next_state = subgame + (best_proposal, )
        result, _ = ACTION_TABLE[next_state]
        ACTION_TABLE[subgame] = (result, [next_state])

    else:
        if player_knows_bad_guy(2, subgame):
            bad_guys = [subgame[BAD]]
        else:
            bad_guys = [0, 1]

        payoff = 0.0
        next_states = []
        for bad_guy in bad_guys:
            proposal = tuple(sorted([2, list({0, 1} - {bad_guy})[0]]))
            next_state = subgame + (proposal,)
            result, _ = ACTION_TABLE[next_state]
            payoff += result * 1.0/len(bad_guys)
            next_states.append(next_state)
        ACTION_TABLE[subgame] = (payoff, next_states)


####
#  Proposal 2    Fail/Succeed
####
for game in GAMES:
    subgame = game[:PROPOSAL_2_SUCCEED]
    if subgame[BAD] not in subgame[PROPOSAL_2]:
        next_state = subgame + (True, )
        result, _ = ACTION_TABLE[next_state]
        ACTION_TABLE[subgame] = (result, [next_state])
        # good guys must succeed a vote. 
    else:
        # bad guy does the thing that minimizes good guy's success chance.
        best_option = min([False, True], key=lambda vote: ACTION_TABLE[subgame + (vote,)][0])
        next_state = subgame + (best_option, )
        result, _ = ACTION_TABLE[next_state]
        ACTION_TABLE[subgame] = (result, [next_state])

####
#  Proposal 2
####

for game in GAMES:
    subgame = game[:PROPOSAL_2]
    if subgame[BAD] == 1:
        # player 3 is the bad guy
        # Play the proposal that minimizes the good guy's chance of success.
        best_proposal = min([(1, 2), (0, 1), (0, 2)], key=lambda proposal: ACTION_TABLE[subgame + (proposal,)][0])
        next_state = subgame + (best_proposal, )
        result, _ = ACTION_TABLE[next_state]
        ACTION_TABLE[subgame] = (result, [next_state])

    else:
        if player_knows_bad_guy(1, subgame):
            bad_guys = [subgame[BAD]]
        else:
            bad_guys = [0, 2]

        payoff = 0.0
        next_states = []
        for bad_guy in bad_guys:
            proposal = tuple(sorted([1, list({0, 2} - {bad_guy})[0]]))
            next_state = subgame + (proposal,)
            result, _ = ACTION_TABLE[next_state]
            payoff += result * 1.0/len(bad_guys)
            next_states.append(next_state)
        ACTION_TABLE[subgame] = (payoff, next_states)


####
#  Proposal 1    Fail/Succeed
####
for game in GAMES:
    subgame = game[:PROPOSAL_1_SUCCEED]
    if subgame[BAD] not in subgame[PROPOSAL_1]:
        next_state = subgame + (True, )
        result, _ = ACTION_TABLE[next_state]
        ACTION_TABLE[subgame] = (result, [next_state])
        # good guys must succeed a vote. 
    else:
        # bad guy does the thing that minimizes good guy's success chance.
        best_option = min([False, True], key=lambda vote: ACTION_TABLE[subgame + (vote,)][0])
        next_state = subgame + (best_option, )
        result, _ = ACTION_TABLE[next_state]
        ACTION_TABLE[subgame] = (result, [next_state])

####
#  Proposal 1
####

for game in GAMES:
    subgame = game[:PROPOSAL_1]
    if subgame[BAD] == 0:
        # player 3 is the bad guy
        # Play the proposal that minimizes the good guy's chance of success.
        best_proposal = min([(0, 1), (0, 2), (1, 2)], key=lambda proposal: ACTION_TABLE[subgame + (proposal,)][0])
        next_state = subgame + (best_proposal, )
        result, _ = ACTION_TABLE[next_state]
        ACTION_TABLE[subgame] = (result, [next_state])

    else:
        if player_knows_bad_guy(0, subgame):
            bad_guys = [subgame[BAD]]
        else:
            bad_guys = [1, 2]

        payoff = 0.0
        next_states = []
        for bad_guy in bad_guys:
            proposal = tuple(sorted([0, list({1, 2} - {bad_guy})[0]]))
            next_state = subgame + (proposal,)
            result, _ = ACTION_TABLE[next_state]
            payoff += result * 1.0/len(bad_guys)
            next_states.append(next_state)
        ACTION_TABLE[subgame] = (payoff, next_states)


import random
ROUNDS = ["Proposal 0", "Proposal 0 succeed", "Proposal 1", "Proposal 1 succeed", "Proposal 2", "Proposal 2 succeed", "Endgame"]
#####

state = (0, )
print("======= Nash for bad guy 0 =======")
print("Good guys win: {}".format(ACTION_TABLE[state][0]))
print("====== Game play ======")

r = 0
while state is not None:
    _, next_states = ACTION_TABLE[state]
    if len(next_states) == 0:
        break
    if len(next_states) > 1:
        print("Making a random choice!!")
    state = random.choice(next_states)
    print("{}: {}".format(ROUNDS[r], state[-1]))
    r += 1


state = (1, )
print("======= Nash for bad guy 1 =======")
print("Good guys win: {}".format(ACTION_TABLE[state][0]))
print("====== Game play ======")

r = 0
while state is not None:
    _, next_states = ACTION_TABLE[state]
    if len(next_states) == 0:
        break
    if len(next_states) > 1:
        print("Making a random choice!!")
    state = random.choice(next_states)
    print("{}: {}".format(ROUNDS[r], state[-1]))
    r += 1


state = (2, )
print("======= Nash for bad guy 2 =======")
print("Good guys win: {}".format(ACTION_TABLE[state][0]))
print("====== Game play ======")

r = 0
while state is not None:
    _, next_states = ACTION_TABLE[state]
    if len(next_states) == 0:
        break
    if len(next_states) > 1:
        print("Making a random choice!!")
    state = random.choice(next_states)
    print("{}: {}".format(ROUNDS[r], state[-1]))
    r += 1
