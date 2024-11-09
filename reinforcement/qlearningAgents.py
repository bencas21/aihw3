# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *
from featureExtractors import SimpleExtractor

import util

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        self.qValues = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        return self.qValues[(state, action)]


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        legal = self.getLegalActions(state)
        if len(legal) == 0:
            return 0.0
        return max(self.getQValue(state, action) for action in legal)

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        actions = self.getLegalActions(state)

        if len(actions) == 0:
            return None

        maxQ = float('-inf')

        optimal = []
        for action in actions:
            qValue = self.getQValue(state, action)
            if qValue > maxQ:
                maxQ = qValue
                optimal = [action]
            elif qValue == maxQ:
                optimal.append(action)
        return random.choice(optimal)  # Randomly select among best actions

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        legalActions = self.getLegalActions(state)
        if len(legalActions) == 0:
            return None

        if util.flipCoin(self.epsilon):
            return random.choice(legalActions)  #
        else:
            return self.computeActionFromQValues(state)

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        nextQ = self.computeValueFromQValues(nextState)
        currentQ = self.getQValue(state, action)
        newQ = currentQ + self.alpha * (reward + self.discount * nextQ - currentQ)
        self.qValues[(state, action)] = newQ

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    def __init__(self, extractor='IdentityExtractor', **args):
        """
        Initialize ApproximateQAgent.

        If no extractor is passed, default to IdentityExtractor.
        """
        # Set up the feature extractor (by default, use IdentityExtractor)
        if extractor == 'IdentityExtractor':
            self.featExtractor = IdentityExtractor()
        else:
            # If another extractor is passed, use that
            self.featExtractor = extractor()

        # Initialize the weights for features
        self.weights = util.Counter()

        # Initialize the parent class (QLearningAgent)
        super().__init__(**args)

    def getQValue(self, state, action):
        """
        Return the Q-value for a state-action pair.
        In Approximate Q-learning, Q-value is the dot product of feature values and weights.
        """
        features = self.featExtractor.getFeatures(state, action)
        return sum([self.weights[feature] * value for feature, value in features.items()])

    def update(self, state, action, nextState, reward):
        """
        Update the weights based on the Q-value update rule.
        """
        features = self.featExtractor.getFeatures(state, action)
        q_value = self.getQValue(state, action)
        max_q_next = self.getMaxQ(nextState)
        error = (reward + self.discount * max_q_next) - q_value

        for feature, value in features.items():
            self.weights[feature] += self.alpha * error * value

    def getAction(self, state):
        return self.getPolicy(state)

    def getPolicy(self, state):
        possible_actions = state.getLegalActions()
        best_action = None
        best_q_value = float("-inf")

        for action in possible_actions:
            q_value = self.getQValue(state, action)
            if q_value > best_q_value:
                best_q_value = q_value
                best_action = action

        return best_action

    def getValue(self, state):
        return self.getQValue(state, self.getPolicy(state))

    def getWeights(self):
        """
        Return the weights associated with the features.
        """
        return self.weights
