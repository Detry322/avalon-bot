from agent import Agent
import numpy as np
import itertools as it


def all_actions(possible_actions):
    for moves in it.product(*possible_actions):
        prob = 1.0
        for player, move in enumerate(moves):
            p, _ = possible_actions[player][move]
            prob *= p
        yield (moves, prob)

def observe_playthrough_verbose(solver, playthrough, k=2):
    belief = np.ones(len(solver.game.HIDDEN_STATES))
    belief /= np.sum(belief)
    particles = solver.my_particles
    for state, next_state, observation in playthrough:
        print " =========== "
        print " State: {}".format(state)
        print " Belief: {}".format(belief)
        print " Next : {}".format(next_state)
        print " Obs  : {}".format(observation)
        new_particles = solver.generate_and_prune_new_particles(solver.game, state, observation, k, particles)
        print " New Particles:"
        for particle in new_particles:
            prettyprint(particle)
        belief = solver.get_belief_given_particles(solver.game, new_particles)
        print " New Belief: {}".format(belief)
        if k >=2 :
            for player in range(solver.game.NUM_PLAYERS):
                player_belief = solver.get_belief_given_particles(solver.game, new_particles[0].TOM.thoughts[player])
                print " \t TOM player {0} belief: {1}".format(player, player_belief)
                print " \t--------TOM PARTICLES---------"
                for particle in new_particles[0].TOM.thoughts[player]:
                    prettyprint(particle)
        particles = new_particles


    print "Final belief: {}".format(belief)

def observe_playthrough(solver, playthrough, k=2):
    result = []
    belief = np.ones(len(solver.game.HIDDEN_STATES))
    belief /= np.sum(belief)
    particles = solver.my_particles
    for state, next_state, observation in playthrough:
        new_particles = solver.generate_and_prune_new_particles(solver.game, state, observation, k, particles)
        belief = solver.get_belief_given_particles(solver.game, new_particles)
        result.append(belief)
        particles = new_particles
    return result


def prettyprint(particle):
    print "-------- PARTICLE ---------"
    print "\t Hypothesis:"
    print "\t \t Hidden-state: {}".format(particle.Hypothesis.h)
    print "\t \t Explanation:"
    for rnd, moves in enumerate(particle.Hypothesis.explanation):
        print "\t \t \t Round {0}: {1}".format(rnd, moves)
    print "\t \t Score: {}".format(particle.Hypothesis.score)
    # print "\t \t Theory-of-mind: {}".format(particle.TOM)
