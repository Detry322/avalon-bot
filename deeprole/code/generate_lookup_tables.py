import itertools

NUM_PLAYERS = 5

VIEWPOINT_TO_BAD = [[] for _ in range(NUM_PLAYERS)]

for player, viewpoint_arr in enumerate(VIEWPOINT_TO_BAD):
    viewpoint_arr.append(-1) # first viewpoint is neutral
    
    remaining_players = [p for p in range(NUM_PLAYERS) if p != player]
    for bad_guys in itertools.combinations(remaining_players, 2):
        n = (1 << bad_guys[0]) | (1 << bad_guys[1])
        viewpoint_arr.append(n)

    viewpoint_arr.extend(remaining_players) # Partner in crime if assassin
    viewpoint_arr.extend(remaining_players) # Partner in crime if minion

print """
const int VIEWPOINT_TO_BAD[NUM_PLAYERS][NUM_VIEWPOINTS] = {
    %s
};
""" % (",\n    ".join([
    ("{" + ",  ".join(["{:> 3}".format(v) for v in viewpoint_arr]) + " }")
    for viewpoint_arr in VIEWPOINT_TO_BAD
]))


ASSIGNMENT_TO_VIEWPOINT = []

for assignment in itertools.permutations(range(NUM_PLAYERS), 3):
    merlin, assassin, minion = assignment

    viewpoint = [0] * NUM_PLAYERS

    bad = (1 << assassin) | (1 << minion)
    merlin_viewpoint = 1 + VIEWPOINT_TO_BAD[merlin][1:7].index(bad)
    assassin_viewpoint = 7 + VIEWPOINT_TO_BAD[assassin][7:11].index(minion)
    minion_viewpoint = 11 + VIEWPOINT_TO_BAD[minion][11:15].index(assassin)

    viewpoint[merlin] = merlin_viewpoint
    viewpoint[assassin] = assassin_viewpoint
    viewpoint[minion] = minion_viewpoint
    ASSIGNMENT_TO_VIEWPOINT.append(viewpoint)

print """
const int ASSIGNMENT_TO_VIEWPOINT[NUM_ASSIGNMENTS][NUM_PLAYERS] = {
    %s
};
""" % (",\n    ".join([
    ("{" + ",  ".join(["{:> 3}".format(v) for v in viewpoint_arr]) + " }")
    for viewpoint_arr in ASSIGNMENT_TO_VIEWPOINT
]))


VIEWPOINT_TO_PARTNER_VIEWPOINT = [[] for _ in range(NUM_PLAYERS)]

for player, viewpoint_to_bad in enumerate(VIEWPOINT_TO_BAD):
    for viewpoint_index, partner in enumerate(viewpoint_to_bad):
        if viewpoint_index < 7:
            VIEWPOINT_TO_PARTNER_VIEWPOINT[player].append(-1)
            continue

        is_assassin = (viewpoint_index < 11)

        if is_assassin:
            partner_viewpoint = 11 + VIEWPOINT_TO_BAD[partner][11:15].index(player)
        else:
            partner_viewpoint = 7 + VIEWPOINT_TO_BAD[partner][7:11].index(player)

        VIEWPOINT_TO_PARTNER_VIEWPOINT[player].append(partner_viewpoint)

print """
const int VIEWPOINT_TO_PARTNER_VIEWPOINT[NUM_PLAYERS][NUM_VIEWPOINTS] = {
    %s
};
""" % (",\n    ".join([
    ("{" + ",  ".join(["{:> 3}".format(v) for v in viewpoint_arr]) + " }")
    for viewpoint_arr in VIEWPOINT_TO_PARTNER_VIEWPOINT
]))
