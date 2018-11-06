import jinja2
import itertools
from collections import namedtuple

from multiprocessing.pool import ThreadPool

Round = namedtuple('Round', ['number', 'proposal', 'result'])

loader = jinja2.FileSystemLoader(searchpath="./")
environment = jinja2.Environment(loader=loader)

template = environment.get_template('base.template')

games = set([])

players = range(1, 4)
results = ['success', 'fail']
print "generating games"
for bad in players:
    for proposal1 in itertools.combinations(players, 2):
        for result1 in results:
            for proposal2 in itertools.combinations(players, 2):
                for result2 in results:
                    for proposal3 in itertools.combinations(players, 2):
                        for result3 in results:
                            if result1 == 'fail' and bad not in proposal1:
                                continue
                            if result2 == 'fail' and bad not in proposal2:
                                continue
                            if result3 == 'fail' and bad not in proposal3:
                                continue

                            game = (
                                Round(1, proposal1, result1),
                                Round(2, proposal2, result2),
                                Round(3, proposal3, result3),
                            )

                            games.add(game)


sorted_games = sorted(games)

import errno    
import os
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

print "generating html"
for game_id, game in enumerate(sorted_games):
    mkdir_p('games/{}'.format(game_id))
    with open('games/{}/{}.html'.format(game_id, game_id), 'w') as f:
        f.write(template.render(remarks='', rounds=game))


print 'generating pdfs'

import pdfkit
def generate_pdf(game_id):
    print "generating {}".format(game_id)
    pdfkit.from_file('games/{}/{}.html'.format(game_id,game_id), 'games/{}/{}.pdf'.format(game_id,game_id), options={'page-width': 200, 'page-height': 120})

pool = ThreadPool()
pool.map(generate_pdf, range(len(sorted_games)))
# Basic.html

# rounds = [
#     Round(1, [1, 2], 'fail'),
#     Round(2, [2, 3], 'fail'),
#     Round(3, [1, 3], 'success'),
# ]
# remarks = "Player 2 is bad, will fail every mission he's on. Proposal size: {}".format(len(rounds[0].proposal))

# Basic2.html
# rounds = [
#     Round(1, [1, 2], 'fail'),
#     Round(2, [2, 3], 'success'),
#     Round(3, [2, 3], 'success'),
# ]
# remarks = "Player 1 is bad, will fail every mission he's on."

# What to propose on round 3?
# rounds = [
#     Round(1, [1, 2], 'fail'),
#     Round(2, [2, 3], 'success'),
#     Round(3, [2, 3], 'fail'),
# ]
# remarks = "Who should player 3 propose?"

# How many bad people?
# rounds = [
#     Round(1, [1, 4], 'success'),
#     Round(2, [2, 4], 'fail'),
#     Round(3, [2, 3], 'fail'),
#     Round(4, [3, 4], 'fail'),
# ]
# remarks = "Who's bad in this case? Are they skilled? How many bad people are there?"


# rounds = [
#     Round(1, [2, 3], 'fail'),
#     Round(2, [1, 2], 'success'),
#     Round(3, [1, 3], 'success'),
#     Round(4, [1, 2], 'fail'),
# ]
# remarks = "One bad guy, who's bad at every timestep?"


# rounds = [
#     Round(1, [1, 2], 'fail'),
#     Round(2, [1, 2], 'success'),
#     Round(3, [1, 3], 'success'),
# ]
# remarks = "Who's bad at every timestep?"

# rounds = [
#     Round(1, [2, 3], 'fail'),
#     Round(2, [1, 2], 'success'),
#     Round(3, [1, 3], 'success'),
#     Round(4, [1, 4], 'success'),
# ]
# remarks = "One bad guy, who's bad at every timestep?"
