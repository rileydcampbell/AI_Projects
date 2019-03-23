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

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
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
        for i in range(self.iterations):
            updatedVals = util.Counter()
            for state in self.mdp.getStates():
                if not self.mdp.isTerminal(state):
                    vals=-10000000000000000
                    for action in self.mdp.getPossibleActions(state):
                        vals = max(self.computeQValueFromValues(state, action), vals)
                        updatedVals[state] = vals
            self.values = updatedVals


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
        val = 0
        for info in self.mdp.getTransitionStatesAndProbs(state, action):
            p = info[1]
            nextState = info[0]
            reward = self.mdp.getReward(state, action, nextState)
            decay = self.discount*(self.values[nextState])
            val += p*(reward+decay)
        return val



    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        action_val = util.Counter()
        action_val[None] = -1000000000000000000000000
        if self.mdp.isTerminal(state):
            return None

        for action in self.mdp.getPossibleActions(state):
            action_val[action] = self.computeQValueFromValues(state,action)
        return action_val.argMax()






    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
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
        for i in range(self.iterations):
            # updatedVals = util.Counter()
            state_to_update =  self.mdp.getStates()[i % len(self.mdp.getStates())]
            if self.mdp.isTerminal(state_to_update):
                a=0
            else:
                vals=-1000000000000000
                for action in self.mdp.getPossibleActions(state_to_update):
                    vals = max(self.computeQValueFromValues(state_to_update, action), vals)
                self.values[state_to_update] = vals


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
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
        predecessors = dict()
        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                for action in self.mdp.getPossibleActions(state):
                    for newState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                        if (newState in predecessors):
                            predecessors[newState].append(state)
                        else:
                            predecessors[newState] = []
        for state in predecessors:
            predecessors[state] = set(predecessors[state])
        priority_queue = util.PriorityQueue()
        for state in self.mdp.getStates():
            if (not self.mdp.isTerminal(state)):
                for action in self.mdp.getPossibleActions(state):
                    max_val = -100000
                    max_val = max(self.computeQValueFromValues(state,action),max_val)
                diff = abs(self.values[state]-max_val)
                priority_queue.push(state, -diff)
        for i in range(self.iterations):
            if (priority_queue.isEmpty()):
                break
            else:
                s = priority_queue.pop()
                if (not self.mdp.isTerminal(s)):
                    max_val = -10000
                    for action in self.mdp.getPossibleActions(s):
                        max_val = max(self.computeQValueFromValues(s, action), max_val)
                    self.values[s] = max_val
                for pred in predecessors[s]:
                    if (not self.mdp.isTerminal(pred)):
                        max_val = -10000
                        for actions in self.mdp.getPossibleActions(pred):
                            max_val = max(self.computeQValueFromValues(pred, actions), max_val)
                        diff = abs(self.values[pred]-max_val)
                        if (diff > self.theta):
                            priority_queue.update(pred, -diff)
