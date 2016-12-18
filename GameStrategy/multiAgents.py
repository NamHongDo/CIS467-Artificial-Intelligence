# multiAgents.py
# --------------
# FinalRevision
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
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        walls = successorGameState.getWalls()
        straightlineDistance = ( (walls.width)**2 + (walls.height)**2 )**0.5
        foodList = successorGameState.getFood().asList()

        "*** YOUR CODE HERE ***"
        foodDistance = []#distance of each food
        closestFood = 0#the one food that's closest
        ghostsDistance = []#position of each ghost
        closestGhost = 0#closest ghost that will kill pacman
        
        #Determine if ghosts are edible
        ghostsAreEdible = False
        if newScaredTimes[0]>5:
            ghostsAreEdible = True

        for food in foodList:
            foodDistance.append(manhattanDistance(newPos, food))
        
        #If there is more than one food, return the nearest food
        if len(foodDistance) > 0:
            closestFood = min(foodDistance)
          
        #Make sure not to divide by 0
        if closestFood > 0:
            closestFoodReciprocal = (1.0/closestFood)
        else:
            closestFoodReciprocal = 0
        
        #If ghosts are scared, nearest food becomes most attractive because there is no threat
        if ghostsAreEdible:
            closestFoodReciprocal = closestFoodReciprocal * 10

        for ghost in newGhostStates:
            ghostsDistance.append(manhattanDistance(newPos,ghost.getPosition()))
        
        #If more than one ghost, set closest ghost
        if len(ghostsDistance) > 0:
            closestGhost = min(ghostsDistance)
            #Make sure not to divide by 0
            if closestGhost > 0:
                closestGhostReciprocal = -2.0/closestGhost
            else:
                closestGhostReciprocal = 0
          
        #if closest ghost is far, not as much of a threat
        if closestGhost > straightlineDistance/4.0 and not ghostsAreEdible:
            closestGhostReciprocal = closestGhostReciprocal * -1
        #Otherwise, closest ghost is pretty close, more of a threat
        elif closestGhost < straightlineDistance/5.0 and not ghostsAreEdible:
            closestGhostReciprocal = closestGhostReciprocal * 4
          
        #Ghosts are edible, want to go towards them
        if ghostsAreEdible:
            closestGhostReciprocal = closestGhostReciprocal * -10

        score = successorGameState.getScore()
        weightedScore = score + closestGhostReciprocal + closestFoodReciprocal

        return weightedScore
        #return successorGameState.getScore()

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
    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.
          Here are some method calls that might be useful when implementing minimax.
          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1
          gameState.generateSuccessor(agentIndex, action):
            Returns the successorScoreessor game state after an agent takes an action
          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        depth=0
        agent=0
        listWithMaxAction = self.minimax(gameState,agent,depth) 
        action = listWithMaxAction[1]
        return action

    def minimax(self, gameState,agent,depth):
        if agent == gameState.getNumAgents(): #pacman turn to play, set agent index to 0
            agent=0
        if depth == self.depth * gameState.getNumAgents() or gameState.isWin() or gameState.isLose():         
            return [self.evaluationFunction(gameState),None]
        if agent==0:
            # pacman
            return self.maxvalue(gameState,agent,depth)
        else:
            # ghosts          
            return self.minvalue(gameState,agent,depth)

    def minvalue(self, gameState, agent, depth):
        actions = gameState.getLegalActions(agent)
        minValue = [float("inf"),None]
        for action in actions:
            successorScore = self.minimax(gameState.generateSuccessor(agent,action),agent+1, depth+1) #expanding node from ghost to Pacman
            if minValue[0] > successorScore[0]:
                minValue =[successorScore[0],action] #check and update min value for all actions expanded for ghost
        return minValue

    def maxvalue(self, gameState,agent,depth):
        actions = gameState.getLegalActions(agent)
        maxValue = [-float("inf"),None]
        for action in actions:
            successorScore = self.minimax(gameState.generateSuccessor(agent,action),agent+1, depth+1) #expanding node from pacman to ghost
            if  maxValue[0] < successorScore[0]:
                maxValue = [successorScore[0],action] #check and update max value for all actions expanded for pacman
        return maxValue


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    def getAction(self, gameState):
        agent=0
        depth=0
        alpha=-float("inf")
        beta=float("inf")
        listWithMaxAction = self.alphabeta(gameState,agent,depth,alpha,beta)
        action = listWithMaxAction[1]
        return action

    def alphabeta(self, gameState,agent,depth,alpha,beta):
        if agent == gameState.getNumAgents(): # pacman turn to play, set agent index back to 0
            agent=0
        if depth == self.depth * gameState.getNumAgents() or gameState.isWin() or gameState.isLose():
            return [self.evaluationFunction(gameState),None]
        if agent==0:
            # pacman
            return self.maxvalue(gameState,agent,depth,alpha,beta)
        else:
            # ghosts        
            return self.minvalue(gameState,agent,depth,alpha,beta)

    def minvalue(self, gameState, agent, depth,alpha,beta):
        minValue = [float("inf"),None]
        for action in gameState.getLegalActions(agent):
            successorScore = self.alphabeta(gameState.generateSuccessor(agent,action),agent+1, depth+1,alpha,beta) #expanding node from ghost to pacman
            if minValue[0] > successorScore[0]:
                minValue =[successorScore[0],action]
            if  alpha > minValue[0]:
                return minValue
            beta=min(minValue[0],beta)
        return minValue

    def maxvalue(self, gameState,agent,depth,alpha,beta):           
        maxValue = [-float("inf"),None]
        for action in gameState.getLegalActions(agent):
            successorScore = self.alphabeta(gameState.generateSuccessor(agent,action),agent+1, depth+1,alpha,beta) #expanding node from pacman to ghost
            if  maxValue[0] < successorScore[0]:
                maxValue = [successorScore[0],action]
            if beta < maxValue[0]:
                return maxValue
            alpha=max(maxValue[0],alpha)
        return maxValue


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
        actions = gameState.getLegalActions(0)
        score = -1000
        
        #We will use expectedScore when ghosts are present in the maze
        def expectedScore(gameState, agentIndex, depth):
            numberOfGhosts = gameState.getNumAgents() - 1
            finalScore = 0

            #If game is over
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)
                
            actions = gameState.getLegalActions(agentIndex)    
            numberOfActions = len(actions)
            
            #for each possible action
            for action in actions:
                sucessorState = gameState.generateSuccessor(agentIndex, action)
                #if pacman is the only agent
                if (agentIndex == numberOfGhosts):
                    finalScore += maxScore(sucessorState, depth - 1)
                #otherwise, there are random ghosts
                else:
                    #note: agentIndex + 1 because need to increment agent
                    finalScore += expectedScore(sucessorState, agentIndex + 1, depth)
                    
            return finalScore / numberOfActions
        
        #We will use maxScore when pacman is safe in the maze    
        def maxScore(gameState, depth):
            score = -1000
            
            #If the game is over
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)
            
            actions = gameState.getLegalActions(0)
            #for each possible action
            for action in actions:
                tmpScore = score
                sucessorState = gameState.generateSuccessor(0, action)
                score = max(score, expectedScore(sucessorState, 1, depth))
            return score
        
        #Main method  
        #If game is over    
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
            
        #for each possible action
        for action in actions:
            sucessorState = gameState.generateSuccessor(0, action)
            tmpScore = score
            score = max(score, expectedScore(sucessorState, 1, self.depth))
            
            if score > tmpScore:
                bestAction = action
                
        return bestAction
        
        util.raiseNotDefined() 

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: I implemented the ReflexAgent by determining how much food is present, taking into consideration the distance of the food, the distance of the ghosts, and whether or not the ghosts are scared, and I implemented an agent that will perform well after considering it's given situation.
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    walls = currentGameState.getWalls()
    straightlineDistance = ( (walls.width)**2 + (walls.height)**2 )**0.5
    foodList = currentGameState.getFood().asList()
    foodDistance = []#distance of each food
    closestFood = 0#the one food that's closest
    ghostsDistance = []#position of each ghost
    closestGhost = 0#closest ghost that will kill pacman
    
    #Determine if ghosts are edible
    ghostsAreEdible = False
    #Determines how many moves are left before ghosts stop being edible, sets variable
    if newScaredTimes[0]>5:
        ghostsAreEdible = True

    for food in foodList:
        foodDistance.append(manhattanDistance(newPos, food))
    
    #If there is more than one food, return the nearest food
    if len(foodDistance) > 0:
        closestFood = min(foodDistance)
      
    #Make sure not to divide by 0
    if closestFood > 0:
        closestFoodReciprocal = (1.0/closestFood)
    else:
        closestFoodReciprocal = 0
    
    #If ghosts are scared, nearest food becomes most attractive because there is no threat
    if ghostsAreEdible:
        closestFoodReciprocal = closestFoodReciprocal * 10

    for ghost in newGhostStates:
        ghostsDistance.append(manhattanDistance(newPos,ghost.getPosition()))
    
    #If more than one ghost, set closest ghost
    if len(ghostsDistance) > 0:
        closestGhost = min(ghostsDistance)
        #Make sure not to divide by 0
        if closestGhost > 0:
            closestGhostReciprocal = -2.0/closestGhost
        else:
            closestGhostReciprocal = 0
      
    #if closest ghost is far, not as much of a threat
    if closestGhost > straightlineDistance/4.0 and not ghostsAreEdible:
        closestGhostReciprocal = closestGhostReciprocal * -1
    #Otherwise, closest ghost is pretty close, more of a threat
    elif closestGhost < straightlineDistance/5.0 and not ghostsAreEdible:
        closestGhostReciprocal = closestGhostReciprocal * 4
      
    #Ghosts are edible, want to go towards them
    if ghostsAreEdible:
        closestGhostReciprocal = closestGhostReciprocal * -10

    score = currentGameState.getScore()
    weightedScore = score + closestGhostReciprocal + closestFoodReciprocal

    return weightedScore
    #return successorGameState.getScore()
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

