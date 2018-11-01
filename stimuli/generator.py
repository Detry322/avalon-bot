import jinja2
from collections import namedtuple

Round = namedtuple('Round', ['number', 'proposal', 'result'])

loader = jinja2.FileSystemLoader(searchpath="./")
environment = jinja2.Environment(loader=loader)

template = environment.get_template('base.template')

rounds = [
    Round(1, [2, 3], 'fail'),
    Round(2, [1, 2], 'success'),
    Round(3, [1, 3], 'success'),
    Round(4, [1, 2], 'fail'),
]
remarks = "One bad guy, who's bad at every timestep?"

with open('output.html', 'w') as f:
    f.write(template.render(remarks=remarks, rounds=rounds))


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
