# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util
import heapq

from learningAgents import ValueEstimationAgent
from util import PriorityQueue
import collections

class ValueIterationAgent(ValueEstimationAgent):
    # Got help from github copilot 
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for _ in range(self.iterations):
            new_values = util.Counter()
            for state in self.mdp.getStates():
                if self.mdp.isTerminal(state):
                    continue
                max_value = float('-inf')

                for action in self.mdp.getPossibleActions(state):
                    value = self.computeQValueFromValues(state, action)
                    max_value = max(max_value, value)
                new_values[state] = max_value
            self.values = new_values


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        q = 0
        for state_next, probability in self.mdp.getTransitionStatesAndProbs(state, action):
            r = self.mdp.getReward(state, action, state_next)
            q += probability * (r + (self.discount * self.values[state_next]))
        return q
        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        best_action = None
        best_value = float('-inf')
        for action in self.mdp.getPossibleActions(state):
            value = self.computeQValueFromValues(state, action)
            if value > best_value:
                best_value = value
                best_action = action
        return best_action
        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    # Got help from github copilot
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        for _ in range(self.iterations):
            state = states[_ % len(states)]
            if self.mdp.isTerminal(state):
                continue
            max_value = float('-inf')
            for action in self.mdp.getPossibleActions(state):
                value = self.computeQValueFromValues(state, action)
                max_value = max(max_value, value)
            self.values[state] = max_value


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    # Got help from github copilot
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        predecessors = self.initializePredecessors()
        queue = util.PriorityQueue()

        for state in self.mdp.getStates():
            self.updateQueueForState(state, queue)

        for _ in range(self.iterations):
            if queue.isEmpty():
                break

            state = queue.pop()
            if not self.mdp.isTerminal(state):
                max_value = float('-inf')
                for action in self.mdp.getPossibleActions(state):
                    value = self.computeQValueFromValues(state, action)
                    max_value = max(max_value, value)
                self.values[state] = max_value

            for predecessor in predecessors[state]:
                self.updateQueueForState(predecessor, queue)

    def initializePredecessors(self):
        predecessors = {state: set() for state in self.mdp.getStates()}

        for state in self.mdp.getStates():
            for action in self.mdp.getPossibleActions(state):
                for next_state, _ in self.mdp.getTransitionStatesAndProbs(state, action):
                    predecessors[next_state].add(state)

        return predecessors

    def updateQueueForState(self, state, queue):
        if not self.mdp.isTerminal(state):
            max_value = float('-inf')

            for action in self.mdp.getPossibleActions(state):
                value = self.computeQValueFromValues(state, action)
                max_value = max(max_value, value)

            diff = abs(self.values[state] - max_value)
            queue.update(state, -diff)