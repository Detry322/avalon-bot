import json


class Game(object):
    spies_win = False
    start = ""
    end = ""

    """Docstring for Game. """
    def __init__(self, id):
        """TODO: to be defined1. """
        self.players = []
        self.id = id
        self.special_role_data = {}
        self.role_data = None
        self.log = None
        self.id_map = {}

    def add_player(self, player):
        self.players.append(player)

    def canonicalize_role(self):
        for p in self.players:
            self.id_map[p["player_id"]] = p["seat"]
        self.role_data = {}
        data = self.special_role_data.items()
        for k, v in data:
            if k == "leader":
                self.role_data[k] = v
                continue
            self.role_data[k] = self.id_map[v]
        self.special_role_data = None

    def serialize(self):
        out = {
            "id": self.id,
            "spies_win": self.spies_win,
            "start": self.start,
            "end": self.end,
            "roles": self.role_data,
            "players": sorted(self.players, key=lambda x: x["seat"]),
            "log": self.log,
        }
        return out


def create_games(filename):
    out = {}
    f = open(filename)
    for l in f.readlines():
        l = l.strip()
        fields = l.split("\t")
        id = int(fields[0])
        game = Game(id)
        game.start = fields[2]
        game.end = fields[3]
        game.spies_win = bool(int(fields[4]))
        game.special_role_data = json.loads(fields[1])
        out[id] = game
    f.close()
    return out


def load_players(gameset, filename):
    f = open(filename)
    for l in f.readlines():
        l = l.strip()
        fields = l.split("\t")
        id = int(fields[0])
        player = {
            "seat": int(fields[1]),
            "player_id": int(fields[2]),
            "spy": bool(int(fields[3])),
        }
        gameset[id].add_player(player)

    f.close()

def replay_game(game, log):
    size = len(game.players)
    out = []
    round_n = 1
    vote_n = 1
    wins = 0
    state = "propose"
    vote = {}
    round = {}
    for entry in log:
        data = json.loads(entry[2])
        if data["cmd"] != "choosePlayers" and data["cmd"] != "choose":
            continue
        if state == "propose":
            if data["cmd"] != "choosePlayers":
                continue
                raise Exception("Invalid propose State:" + data["cmd"] + ":" + str(game.id) + ":" + entry[0])
            vote = {"proposer": game.id_map[int(entry[1])],
                    "votes" : [None] * size,
                    "team": sorted([game.id_map[x] for x in data["choice"]])
                   }
            state = "vote"
        elif state == "vote":
            if data["cmd"] != "choose":
                continue
                raise Exception("Invalid vote State:" + data["cmd"] + ":" + str(game.id) + ":" + entry[0])
            if data["choice"] != "Approve" and data["choice"] != "Reject":
                continue
            if int(entry[1]) not in game.id_map:
                continue
            vote["votes"][game.id_map[int(entry[1])]] = data["choice"]
            if len([x for x in vote["votes"] if x is None]) == 0:
                round[vote_n] = vote
                if len([x for x in vote["votes"] if x == "Approve"]) > len([x for x in vote["votes"] if x == "Reject"]):
                    round["mission"] = [None] * len(vote["team"])
                    state = "mission"
                else:
                    vote = {}
                    vote_n += 1
                    state = "propose"
        elif state == "mission":
            if data["cmd"] != "choose":
                continue
                raise Exception("Invalid mission State:" + data["cmd"] + ":" + str(game.id) + ":" + entry[0])
            for i, x in enumerate(vote["team"]):
                if game.id_map[int(entry[1])] == x:
                    round["mission"][i] = data["choice"]
                    break
            if len([x for x in round["mission"] if x is None]) == 0:
                vote_n = 1
                round_n += 1
                if len([x for x in round["mission"] if x == "Fail"]) == 0:
                    wins += 1
                if wins == 3 and "assassin" in game.role_data and "merlin" in game.role_data:
                    state = "find_merlin"
                elif "ladyOfTheLake" in game.role_data and round_n >= 3:
                    state = "lol"
                else:
                    out.append(round)
                    round = {}
                    state = "propose"
        elif state == "lol":
            if data["cmd"] != "choosePlayers":
                continue
                raise Exception("Invalid lol State:" + data["cmd"] + ":" + str(game.id) + ":" + entry[0])
            round["ladyOfTheLake"] = {
                "from": game.id_map[int(entry[1])],
                "to": game.id_map[data["choice"][0]],
            }
            out.append(round)
            round = {}
            state = "propose"
        elif state == "find_merlin":
            if data["cmd"] != "choosePlayers":
                continue
                raise Exception("Invalid lol State:" + data["cmd"] + ":" + str(game.id) + ":" + entry[0])
            round["findMerlin"] = {
                "assassin": game.id_map[int(entry[1])],
                "merlin_guess": game.id_map[data["choice"][0]],
                "merlin": game.role_data["merlin"],
            }
            out.append(round)
            round = {}
            state = "propose"

    game.log = out


def add_vote_data(gameset, filename):
    f = open(filename)
    current_id = None
    current_log = []
    for l in f.readlines():
        l = l.strip()
        fields = l.split("\t")
        if int(fields[0]) != current_id:
            if current_id is not None:
                replay_game(gameset[current_id], current_log)
            current_id = int(fields[0])
            current_log = []
        current_log.append(fields[1:])
    if current_id is not None:
        replay_game(gameset[current_id], current_log)
    f.close()



gameset = create_games("games.tsv")
load_players(gameset, "gameplayers.tsv")

for x in gameset:
    gameset[x].canonicalize_role()

print("Computing 25000")
add_vote_data(gameset, "gamelog_25000.tsv")
print("Computing 20000")
add_vote_data(gameset, "gamelog_20000.tsv")
print("Computing 15000")
add_vote_data(gameset, "gamelog_15000.tsv")
print("Computing 10000")
add_vote_data(gameset, "gamelog_10000.tsv")
print("Computing 5000")
add_vote_data(gameset, "gamelog_5000.tsv")
print("Computing 0")
add_vote_data(gameset, "gamelog_0.tsv")

print(gameset[32766].serialize())

import json

print "Dumping to games.json"
with open('games.json', 'w') as f:
    games = [game.serialize() for _, game in sorted(gameset.items())]
    json.dump(games, f, indent=2)
