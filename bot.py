

class GoodBot(object):
    def __init__(self, id_, bot_ids, num_bad):
        self.id = id_,
        self.bot_ids = bot_ids
        self.beliefs = {}
        for bot_id in bot_ids:
            pass

    def propose(self, depth=0):
        # Just propose the people you know to be good.
        pass

    def vote(self, proposal, depth=0):
        # Just vote based on the probability you think they're good.
        pass

    def examime_vote_result(self, proposal, votes, depth=0):
        pass

    def examime_round_result(self, proposal, num_fails, depth=0):
        pass


class BadBot(object):
    def __init__(self, id_, bot_ids, bad_bots):
        self.id = id_
        self.bot_ids = bot_ids
        self.bad_bots = bad_bots

    def propose(self, depth=0):
        pass

    def vote(self, depth=0):
        pass

    def examime_vote_result(self, proposal, votes, depth=0):
        # Do nothing, you know everything.
        return

    def examime_round_result(self, proposal, num_fails, depth=0):
        # Do nothing, you know everything.
        return      


class GameState(object):
    pass


gan
