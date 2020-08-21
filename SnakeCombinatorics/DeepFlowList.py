import StateValueNode as SVN
import Combinatorics as comb
import math
import Point
import numpy as np
from copy import copy, deepcopy
class DeepFlowList:
    def __init__(self, size, discountFactor):
        # the size in one direction of a board 
        self.size = size
        self.discountFactor = discountFactor
        # the maximum length of the snake
        self.squaredSize = int(math.pow(size, 2))
        # Stores the entire data structue with columns of SVNs
        self.columnarSVNList = []
        self.SVNDict = {}
        for i in range(self.squaredSize):
            self.columnarSVNList.append([])
        print("starting construction")
        for i in range(self.squaredSize):
            # counts backwards from squaredsize
            currentLength = self.squaredSize - i
            # if this is the last column (states where the snake has won)
            if currentLength == self.squaredSize:
                # in the construction of the DFl, the SVNs for the endgame winning states are created first
                lastColumnStates = comb.getStatesFromSingleSizes(currentLength-1, size)
                for j in lastColumnStates:
                    # because these are possible final states that win the game, their value is assigned as 1
                    node = deepcopy(SVN.StateValueNode(j, 100))
                    self.columnarSVNList[currentLength-1].append(node)
                    self.SVNDict[str(j)] = node
            else:
                states = comb.getStatesFromSingleSizes(currentLength-1, size)
                SVNs = []
                for j in states:
                    nSVN = deepcopy(SVN.StateValueNode(j, -0.001))
                    SVNs.append(nSVN)
                    self.SVNDict[str(j)] = nSVN
                self.columnarSVNList[currentLength-1] = SVNs
        print(self.columnarSVNList)
        print("begin connecting")
        for i in range(self.squaredSize-1):
            print(i)
            # in the construction of the DFl, the SVNs for the endgame winning states are created first
            columnSVNs = self.columnarSVNList[i]
            for j in columnSVNs:
                l, r, u, d = self.getPossibleMoves(j.getState())
                for m in l:
                    j.addToLeft(self.SVNDict[str(m)])
                for m in r:
                    j.addToRight(self.SVNDict[str(m)])
                for m in u:
                    j.addToUp(self.SVNDict[str(m)])
                for m in d:
                    j.addToDown(self.SVNDict[str(m)])

        # rate of change when process stops
        theta = 0.1
        loss = 0.2
        print("begin value it")
        while loss > theta:
            loss = self.bPvalueIteration()
            print("loss " + str(loss))
            for i in self.columnarSVNList:
                # print(i)
                for j in i:
                    # print("------------------")
                    # print("state" + str(j.getState()))
                    a, b, c, d = j.getMoves()
                   

    """takes two states (2d board arrays) and returns 0, 1, 2, 3 (lrud) for which direction the next state 
    is as a move or returns -1 if nextState is not an actual possible next state"""
    def isPossibleNextState(self, state, nextState):
        # if the snakes are of the same length (the snake has not eaten the food from one state to the next)
        if self.getLength(state) == self.getLength(nextState):
            for x in range(len(state)):
                for y in range(len(state[0])):
                    # gets the value of this location
                    num = state[x][y]
                    # if the foodPos is not the same, it is not a possible next state
                    if num == -1:
                        if nextState[x][y] != -1:
                            return -1

                    # since the snake can only move by 1 square each move, the difference 
                    # between all nextPos and currentPos will be 1 (even including end pos bc we r 
                    # testing from nextState to State)
                    elif nextState[x][y] > 0:
                        if nextState[x][y] == 1:
                            if num != 0 and num != self.getLength(num):
                                return -1
                        elif nextState[x][y] - num != 1:
                                return -1
            # since head positions are different, test like this
            head1 = self.getHeadLocation(state)
            head2 = self.getHeadLocation(nextState)
            # if head positions more than one space apart
            if abs(head1.x - head2.x) > 1:
                return -1
            if abs(head1.y - head2.y) > 1:
                return -1
            if abs(head1.x + head1.y - head2.x - head2.y) > 1:
                return -1
            # left
            if head1.y > head2.y:
                return 0
            # right
            if head1.y < head2.y:
                return 1
            # up
            if head1.x > head2.x:
                return 2
            # down
            if head1.x < head2.x:
                return 3
        else:
            # the nextState is 1 length longer so some requirements for it to be a possible next state differ
            for x in range(len(state)):
                for y in range(len(state[0])):
                    # gets the value of this location
                    num = state[x][y]
                    # the foodPos of state must be the headPos of nextState unless there is no food
                    if num == -1 and self.getLength(nextState) != int(math.pow(len(nextState), 2)):
                        if nextState[x][y] != 1:
                            return -1
                    
                    # since the snake can only move by 1 square each move, the difference 
                    # between all nextPos and currentPos will be 1 (even including end pos bc we r 
                    # testing from state to nextState [notice this is flipped from the test with equal lengths])
                    elif num > 0:
                        if num - nextState[x][y] != -1:
                            return -1
            # since the head positions are actually the same, need to look at where the food was before
            # it was eaten
            foodLoc = self.getFoodLocation(state)
            head = self.getHeadLocation(state)
            # left
            if foodLoc.y < head.y:
                return 0
            # right
            if foodLoc.y > head.y:
                return 1
            # up
            if foodLoc.x < head.x:
                return 2
            # down
            if foodLoc.x > head.x:
                return 3
            
        
    """takes a board and returns a point (x, y) where the head is located (the head has a value of 1)"""
    def getHeadLocation(self, board):
        for x in range(len(board)):
            for y in range(len(board[0])):
                if board[x][y] == 1:
                    return Point.Point(x, y)

    """takes a board and returns a point (x, y) where the food is located (the food has a value of -1)"""
    def getFoodLocation(self, board):
        for x in range(len(board)):
            for y in range(len(board[0])):
                if board[x][y] == -1:
                    return Point.Point(x, y)

    def getList(self):
        return self.columnarSVNList
    """takes 2d board and returns integer length of snake"""
    def getLength(self, board):
        return np.amax(board)

    def bPvalueIteration(self):
        loss = 0
        # stats at last index
        for x in reversed(range(len(self.columnarSVNList))):
            currentCol = self.columnarSVNList[x]
            for i in currentCol:
                if i.hasNextStates():
                    l, r, u, d = i.getMoves()
                    sumOfValues = 0
                    numOfPossibleNextStates = 0
                    for j in l:
                        sumOfValues += j.getValue()
                        numOfPossibleNextStates += 1
                    for j in r:
                        sumOfValues += j.getValue()
                        numOfPossibleNextStates += 1
                    for j in u:
                        sumOfValues += j.getValue()
                        numOfPossibleNextStates += 1
                    for j in d:
                        sumOfValues += j.getValue()
                        numOfPossibleNextStates += 1
                    # print(sumOfValues)
                    newValue = self.discountFactor * sumOfValues/numOfPossibleNextStates
                    localLoss = abs(i.getValue() - newValue)
                    # print("new value " + str(newValue))
                    # print("localLoss " + str(localLoss))
                    # print("new")
                    # print(newValue)
                    # print("old")
                    # print(i.getValue())
                    # print(loss)
                    if abs(localLoss) > loss:
                        loss = localLoss
                    i.setValue(newValue)
                    # print(i.getState())
                    # print(i.getValue())
        return loss

    """takes a 2d board and returns four arrays for lrud possible moves"""
    def getPossibleMoves(self, curBoard):
        left = []
        right = []
        up = []
        down = []
        head = self.getHeadLocation(curBoard)
        # left - is head.y bc head stores x, y point but it is actually in r, c format and subtracting from column goes to left
        if head.y-1 >= 0:
            length = self.getLength(curBoard)
            # print("left")
            hasEaten = False
            nextBoard = deepcopy(curBoard)
            if curBoard[head.x][head.y-1] is 0 :
                # moving snake to next position after not eating food
                for x in range(len(nextBoard)):
                    for y in range(len(nextBoard[0])):
                        if nextBoard[x][y] == length:
                            nextBoard[x][y] = 0
                        elif nextBoard[x][y] > 0:
                            nextBoard[x][y] += 1
                nextBoard[head.x][head.y-1] = 1
                left.append(nextBoard)
            elif curBoard[head.x][head.y-1] is -1:
                # moving snake to next position after eating food
                hasEaten = True
                for x in range(len(nextBoard)):
                    for y in range(len(nextBoard[0])):
                        if nextBoard[x][y] == self.getLength(nextBoard):
                            nextBoard[x][y] += 1
                        elif nextBoard[x][y] > 0:
                            nextBoard[x][y] += 1
                        elif nextBoard[x][y] == -1:
                            nextBoard[x][y] = 1
            elif curBoard[head.x][head.y-1] == self.getLength(curBoard) and self.getLength(curBoard) >= 4: #>4 to assure that tail is not the exact next part
                # moving snake to next position after moving to former tail position
                length = self.getLength(nextBoard)
                for x in range(len(nextBoard)):
                    for y in range(len(nextBoard[0])):
                        if nextBoard[x][y] == length:
                            nextBoard[x][y] = 1
                        elif nextBoard[x][y] > 0:
                            nextBoard[x][y] += 1
                left.append(nextBoard)
            if hasEaten:
                # possbile stochastic next states (just different placements of the food)
                if self.getLength(nextBoard) != math.pow(len(nextBoard), 2):
                    ps = comb.getPossibleFoodStates(nextBoard)
                    for p in ps:
                        left.append(p)
                else:
                    left.append(nextBoard)
                    
        # right - is head.y bc head stores x, y point but it is actually in r, c format and adding to column goes to right
        if head.y+1 < len(curBoard):
            length = self.getLength(curBoard)
            nextBoard = deepcopy(curBoard)
            hasEaten = False
            if curBoard[head.x][head.y+1] is 0:
                # moving snake to next position after not eating food
                for x in range(len(nextBoard)):
                    for y in range(len(nextBoard[0])):
                        if nextBoard[x][y] == length:
                            nextBoard[x][y] = 0
                        elif nextBoard[x][y] > 0:
                            nextBoard[x][y] += 1
                nextBoard[head.x][head.y+1] = 1
                right.append(nextBoard)
            elif curBoard[head.x][head.y+1] is -1:
                # moving snake to next position after eating food
                hasEaten = True
                for x in range(len(nextBoard)):
                    for y in range(len(nextBoard[0])):
                        # if is tail position
                        if nextBoard[x][y] == self.getLength(nextBoard):
                            nextBoard[x][y] += 1
                        elif nextBoard[x][y] > 0:
                            nextBoard[x][y] += 1
                        elif nextBoard[x][y] == -1:
                            nextBoard[x][y] = 1
            elif curBoard[head.x][head.y+1] == self.getLength(curBoard) and self.getLength(curBoard) >= 4: #>4 to assure that tail is not the exact next part
                # moving snake to next position after moving to former tail position
                length = self.getLength(nextBoard)
                for x in range(len(nextBoard)):
                    for y in range(len(nextBoard[0])):
                        if nextBoard[x][y] == length:
                            nextBoard[x][y] = 1
                        elif nextBoard[x][y] > 0:
                            nextBoard[x][y] += 1
                right.append(nextBoard)
            if hasEaten:
                # possible stochastic next states (just different placements of the food)
                if self.getLength(nextBoard) != math.pow(len(nextBoard), 2):
                    ps = comb.getPossibleFoodStates(nextBoard)
                    for p in ps:
                        right.append(p)
                else:
                    right.append(nextBoard)

        # up - is head.x bc head stores x, y point but it is actually in r, c format and subtracting from row goes up
        if head.x-1 >= 0:
            length = self.getLength(curBoard)
            nextBoard = deepcopy(curBoard)
            # print("up")
            hasEaten = False
            if curBoard[head.x-1][head.y] is 0:
                # moving snake to next position after not eating food
                
                for x in range(len(nextBoard)):
                    for y in range(len(nextBoard[0])):
                        if nextBoard[x][y] == length:
                            nextBoard[x][y] = 0
                        elif nextBoard[x][y] > 0:
                            nextBoard[x][y] += 1
                nextBoard[head.x-1][head.y] = 1
                up.append(nextBoard)
            elif curBoard[head.x-1][head.y] is -1:
                
                # moving snake to next position after eating food
                hasEaten = True
                for x in range(len(nextBoard)):
                    for y in range(len(nextBoard[0])):
                        if nextBoard[x][y] == self.getLength(nextBoard):
                            nextBoard[x][y] += 1
                        elif nextBoard[x][y] > 0:
                            nextBoard[x][y] += 1
                        elif nextBoard[x][y] == -1:
                            nextBoard[x][y] = 1
            elif curBoard[head.x-1][head.y] == self.getLength(curBoard) and self.getLength(curBoard) >= 4: #>4 to assure that tail is not the exact next part
                # moving snake to next position after moving to former tail position
                length = self.getLength(nextBoard)
                for x in range(len(nextBoard)):
                    for y in range(len(nextBoard[0])):
                        if nextBoard[x][y] == length:
                            nextBoard[x][y] = 1
                        elif nextBoard[x][y] > 0:
                            nextBoard[x][y] += 1
                up.append(nextBoard)

            if hasEaten:
                # possbile stochastic next states (just different placements of the food)
                if self.getLength(nextBoard) != math.pow(len(nextBoard), 2):
                    ps = comb.getPossibleFoodStates(nextBoard)
                    for p in ps:
                        up.append(p)
                else:
                    up.append(nextBoard)

        # down - is head.x bc head stores x, y point but it is actually in r, c format and adding tp column goes down
        if head.x+1 < len(curBoard[0]):
            length = self.getLength(curBoard)
            nextBoard = deepcopy(curBoard)
            # print("down")
            hasEaten = False
            if curBoard[head.x+1][head.y] is 0:
                # moving snake to next position after not eating food
                for x in range(len(nextBoard)):
                    for y in range(len(nextBoard[0])):
                        if nextBoard[x][y] == length:
                            nextBoard[x][y] = 0
                        elif nextBoard[x][y] > 0:
                            nextBoard[x][y] += 1
                nextBoard[head.x+1][head.y] = 1
                down.append(nextBoard)
            elif curBoard[head.x+1][head.y] is -1:
                # moving snake to next position after eating food
                hasEaten = True
                for x in range(len(nextBoard)):
                    for y in range(len(nextBoard[0])):
                        if nextBoard[x][y] == self.getLength(nextBoard):
                            nextBoard[x][y] += 1
                        elif nextBoard[x][y] > 0:
                            nextBoard[x][y] += 1
                        elif nextBoard[x][y] == -1:
                            nextBoard[x][y] = 1
            elif curBoard[head.x+1][head.y] == self.getLength(curBoard) and self.getLength(curBoard) >= 4: #>4 to assure that tail is not the exact next part
                # moving snake to next position after moving to former tail position
                length = self.getLength(nextBoard)
                for x in range(len(nextBoard)):
                    for y in range(len(nextBoard[0])):
                        if nextBoard[x][y] == length:
                            nextBoard[x][y] = 1
                        elif nextBoard[x][y] > 0:
                            nextBoard[x][y] += 1
                down.append(nextBoard)

            if hasEaten:
                # possible stochastic next states (just different placements of the food)
                if self.getLength(nextBoard) != math.pow(len(nextBoard), 2):
                    ps = comb.getPossibleFoodStates(nextBoard)
                    for p in ps:
                        down.append(p)
                else:
                    down.append(nextBoard)

        return left, right, up, down



# e = DeepFlowList(3, 0.9)
# # print(e.getLength([[0, 0, -1], [5, 6, 1], [4, 3, 2]]))
# print(e.isPossibleNextState([[0, 0, -1], [4, 5, 6], [3, 2, 1]], [[0, 0, -1], [5, 6, 1], [4, 3, 2]]))
# # # # # print(e.isPossibleNextState([[1, 2, 0], [4, 3, 0], [0, 0, -1]]))
