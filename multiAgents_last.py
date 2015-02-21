# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
from game import Agent
from game import Actions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        oldFood = currentGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        newCap = successorGameState.getCapsules()
        oldCap = currentGameState.getCapsules()
        BIGNUM = 999999
        score = 0;
        if successorGameState.isLose():
          return -BIGNUM
        if successorGameState.isWin():
          return BIGNUM

        shortestFood = BIGNUM;
        for p in newFood.asList():
          d = util.manhattanDistance(newPos, p)
          if d < shortestFood:
            shortestFood = d
        score -= d
        
        for p in newCap:
          d = util.manhattanDistance(newPos, p)
          score -= 10 * d

        
        for g in newGhostStates:
          p = g.getPosition()
          d = util.manhattanDistance(newPos,p)
          if g.scaredTimer > 0: # not scared 
            score -= d
          else:  # scared 
            if d < 15:
              score += d
        nfoodlist = len(newFood.asList())
        ofoodlist = len(oldFood.asList())
        ncaplist = len(newCap)
        ocaplist = len(oldCap)
        # score = score - 50*(nfoodlist - ofoodlist)
        score = score - 50*(nfoodlist - ofoodlist) - 100*(ncaplist - ocaplist)

        "*** YOUR CODE HERE ***"
        return score

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game

          gameState.isWin():
            Returns whether or not the game state is a winning state

          gameState.isLose():
            Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # print("number of agents: ", gameState.getNumAgents(), "depth: ", self.depth)
        return self.value(gameState, self.index, self.depth)[1]
      
    def value(self,gameState,agentIndex, currdepth):
      """ an evaluate function that returns the value of current node"""
      # print("agentIndex: ", agentIndex, "currdepth: ", currdepth)
      BIGNUM = float("inf")
      numofghosts = gameState.getNumAgents()- 1 
      
      if currdepth == 0 or gameState.isLose() or gameState.isWin():
        finalvalue = self.evaluationFunction(gameState)
        return (finalvalue, None)
      if agentIndex == 0: #maximizer
        bestvalue = -BIGNUM
        bestaction = None
        for possibleaction in gameState.getLegalActions(agentIndex):
          successorState = gameState.generateSuccessor(agentIndex, possibleaction)
          val, action = self.value(successorState,agentIndex + 1, currdepth)
          if val > bestvalue:
            bestvalue = val
            bestaction = possibleaction
        return (bestvalue, bestaction)
      else:
        bestvalue = BIGNUM
        bestaction = None 
        for possibleaction in gameState.getLegalActions(agentIndex):
          successorState = gameState.generateSuccessor(agentIndex, possibleaction)
          if agentIndex < numofghosts:
            val, action = self.value(successorState, agentIndex + 1, currdepth )
          else:
            val, action = self.value(successorState, 0, currdepth - 1 )
          if val < bestvalue:
            bestvalue = val
            bestaction = possibleaction
        return (bestvalue, bestaction)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        BIGNUM = float("inf")
        return self.value(gameState, self.index, self.depth, -BIGNUM, BIGNUM)[1]


    def value(self,gameState,agentIndex, currdepth, alpha, beta):
      """ an evaluate function that returns the value of current node
      alpha is the max's best option on path to root
      beta is min's best option on path to root """
      # print("agentIndex: ", agentIndex, "currdepth: ", currdepth)
      BIGNUM = float("inf")
      numofghosts = gameState.getNumAgents()- 1 
      
      if currdepth == 0 or gameState.isLose() or gameState.isWin():
        finalvalue = self.evaluationFunction(gameState)
        return (finalvalue, None)
      if agentIndex == 0: #maximizer
        bestvalue = -BIGNUM
        bestaction = None
        for possibleaction in gameState.getLegalActions(agentIndex):
          successorState = gameState.generateSuccessor(agentIndex, possibleaction)
          val, action = self.value(successorState,agentIndex + 1, currdepth, alpha, beta)
          if val > bestvalue:
            bestvalue = val
            bestaction = possibleaction
          if bestvalue > beta:
            return (bestvalue, bestaction)
          alpha = max(alpha, bestvalue)
        return (bestvalue, bestaction)
      else: #minimizer 
        bestvalue = BIGNUM
        bestaction = None 
        for possibleaction in gameState.getLegalActions(agentIndex):
          successorState = gameState.generateSuccessor(agentIndex, possibleaction)
          if agentIndex < numofghosts:
            val, action = self.value(successorState, agentIndex + 1, currdepth, alpha, beta)
          else:
            val, action = self.value(successorState, 0, currdepth - 1, alpha, beta )
          if val < bestvalue:
            bestvalue = val
            bestaction = possibleaction
          if bestvalue < alpha:
            return (bestvalue, bestaction)
          beta = min(beta, bestvalue) 
        return (bestvalue, bestaction)

        

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.value(gameState, self.index, self.depth)[1]
      
    def value(self,gameState,agentIndex, currdepth):
      """ an evaluate function that returns the value of current node"""
      # print("agentIndex: ", agentIndex, "currdepth: ", currdepth)
      BIGNUM = float("inf")
      numofghosts = gameState.getNumAgents()- 1 
      
      if currdepth == 0 or gameState.isLose() or gameState.isWin():
        finalvalue = self.evaluationFunction(gameState)
        return (finalvalue, None)
      if agentIndex == 0: #maximizer
        bestvalue = -BIGNUM
        bestaction = None
        for possibleaction in gameState.getLegalActions(agentIndex):
          successorState = gameState.generateSuccessor(agentIndex, possibleaction)
          val, action = self.value(successorState,agentIndex + 1, currdepth)
          if val > bestvalue:
            bestvalue = val
            bestaction = possibleaction
        return (bestvalue, bestaction)
      else: #expectimax
        average = 0
        bestaction = None 
        numofcandidates = float(len(gameState.getLegalActions(agentIndex)))
        for possibleaction in gameState.getLegalActions(agentIndex):
          successorState = gameState.generateSuccessor(agentIndex, possibleaction)
          if agentIndex < numofghosts:
            val, action = self.value(successorState, agentIndex + 1, currdepth )
          else:
            val, action = self.value(successorState, 0, currdepth - 1 )
          average += 1.0/ numofcandidates * val
            
        # bestaction = possibleaction
        return (average, None)

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
      features:
      1. number of foods left
      2. distance(pacman, closest_dots)
      3. distance(pacman, ghosts) 

      linear combination ...
    """
    "*** YOUR CODE HERE ***"
    BIGNUM = 99999
    pacPos = currentGameState.getPacmanPosition()
    foodPos = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates() # in each state, show ghost scared or not
    # scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates] #how much scared time they have left
    capPos = currentGameState.getCapsules()
    wallPos = currentGameState.getWalls()

    numberFoodLeft = len(foodPos.asList())
    numberCapLeft = len(capPos)
    numberDotLeft = numberFoodLeft + numberCapLeft
    score = 0

    #distance to closeset food
    s1Food = BIGNUM;
    s2Food = BIGNUM;
    s3Food = BIGNUM;
    for p in foodPos.asList():
      d = util.manhattanDistance( pacPos, p)
      if d < s1Food:
        s3Food = s2Food
        s2Food = s1Food
        s1Food = d

    if numberFoodLeft == 1:
      prob = PositionSearchProblem(currentGameState, foodPos.asList()[0])
      moves = breadthFirstSearch(prob)
      s1Food = len(moves)
      #print(s1Food)      



    #distance to closest cap
    shortestCap= BIGNUM;
    for p in capPos:
      d = util.manhattanDistance( pacPos, p)
      if d < shortestCap:
        shortestCap = d

    #ghostdistance score   
    for g in ghostStates:
      p = g.getPosition()
      d = util.manhattanDistance( pacPos, p)
      if g.scaredTimer > 0: #scared 
        score -= 100*d
      else:
        score = score + d

    score = score - 100* numberFoodLeft - 300 * numberCapLeft - 2 * s1Food


    if currentGameState.isLose():
      score = -BIGNUM + currentGameState.getScore()
    if currentGameState.isWin():
      score = BIGNUM + currentGameState.getScore()

    print("here, ",pacPos, score, numberFoodLeft,"cap left, ", numberCapLeft)
    return score 

# Abbreviation
better = betterEvaluationFunction

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    closed = set()
    fringe = util.Queue()
    startstate = [(problem.getStartState(), None, 0), []]
    fringe.push(startstate)
    while(True): 
        if fringe.isEmpty():
            return False
        node = fringe.pop()
        if problem.isGoalState(node[0][0]):
            return node[1] 
        if  node[0][0]  not in closed:
            closed.add(node[0][0]) 
            for triple in problem.getSuccessors(node[0][0]):
                prevlst = node[1]
                s = prevlst[:]
                s.append(triple[1])
                currstate = [triple, s]
                fringe.push(currstate)

class PositionSearchProblem(SearchProblem):
    """
    A search problem defines the state space, start state, goal test, successor
    function and cost function.  This search problem can be used to find paths
    to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    Note: this search problem is fully specified; you should NOT change it.
    """

    def __init__(self, gameState, goal, costFn = lambda x: 1, start=None, warn=True, visualize=False):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        if start != None: self.startState = start
        self.goal = goal
        self.costFn = costFn
        self.visualize = visualize
        if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
            print 'Warning: this does not look like a regular search maze'

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        isGoal = state == self.goal

        # For display purposes only
        if isGoal and self.visualize:
            self._visitedlist.append(state)
            import __main__
            if '_display' in dir(__main__):
                if 'drawExpandedCells' in dir(__main__._display): #@UndefinedVariable
                    __main__._display.drawExpandedCells(self._visitedlist) #@UndefinedVariable

        return isGoal

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append( ( nextState, action, cost) )

        # Bookkeeping for display purposes
        self._expanded += 1 # DO NOT CHANGE
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions. If those actions
        include an illegal move, return 999999.
        """
        if actions == None: return 999999
        x,y= self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x,y))
        return cost
