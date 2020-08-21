import DeepFlowList as DFL
import numpy as np
import Point as Point
import pickle
import sys
class Agent:
    sys.setrecursionlimit(100000000)
    """constructor"""
    def __init__(self, boardSize, discountFactor, pathToDFL = None):
        self.boardSize = boardSize
        self.discountFactor = discountFactor
        if pathToDFL == None:
            self.deepFlowList = DFL.DeepFlowList(boardSize, discountFactor)
            with open('saves/DFL'+str(boardSize), 'wb') as saveData:
                pickle.dump(self.deepFlowList, saveData)
        else:
            with open(pathToDFL, 'rb') as saveData:
                self.deepFlowList = pickle.load(saveData)

    def getNextPossibleSVNs(self, board):
        list = self.deepFlowList.getList()
        snakeLength = self.getLength(board)
        column = snakeLength-1
        for i in list[column]:
            if i.getState() == board:
                return i.getMoves()
        return None

    def decide(self, board):
        # print("deciding")
        list = self.deepFlowList.getList()
        snakeLength = self.getLength(board)
        column = snakeLength-1
        dir = -1
        for i in list[column]:
            # print(i.getState())
            if i.getState() == board:
                # print("found")
                l, r, u, d = i.getMoves()

                # print(l)
                # print(r)
                # print(u)
                # print(d)

                max = -1000
                sum = 0
                if len(l) > 0:
                    # e = 0
                    for j in l:
                        sum += j.getValue()
                    ave = sum / len(l)
                    # print(ave)
                    # print(max)
                    if ave > max:
                        max = ave
                        dir = 0
                    sum = 0
                if len(r) > 0:
                    for j in r:
                        sum += j.getValue()
                    ave = sum / len(r)
                    if ave > max:
                        max = ave
                        dir = 1
                    sum = 0
                if len(u) > 0:
                    for j in u:
                        sum += j.getValue()
                    ave = sum / len(u)
                    if ave > max:
                        max = ave
                        dir = 2
                    sum = 0
                if len(d) > 0:
                    for j in d:
                        sum += j.getValue()
                    ave = sum / len(d)
                    if ave > max:
                        max = ave
                        dir = 3
        return dir

    """takes 2d board and returns integer length of snake"""
    def getLength(self, board):
        return np.amax(board)
            
    """takes a board and returns a point (x, y) where the head is located (the head has a value of 1)"""
    def getHeadLocation(self, board):
        for x in range(len(board)):
            for y in range(len(board[0])):
                if board[x][y] == 1:
                    return Point.Point(x, y)

# a = Agent(2, 0.9)
# print(a.decide([[1, 0], [0, -1]]))
# print(a.getNextPossibleSVNs([[0, 0, -1], [1, 0, 0], [0, 0, 0]]))
# board = [[-1, 2, 3], [8, 1, 4], [7, 6, 5]]
# print("----------")
# for i in a.getNextPossibleSVNs(board):
#     print("---" + str(i))
#     for j in i:
#         print(j.getState())
#         print(j.getValue())
#         print(j.down[0].getValue())

# a = Agent(3, 0.9)
# pos = [[-1, 0, 0], [0, 0, 0], [0, 1, 0]]
# # # # # # # next = [[0, -1, 1, 2], [6, 5, 4, 3], [7, 8, 9, 0], [0, 0, 0, 0]]
# # print(a.getNextPossibleSVNs([[-1, 0, 0], [1, 2, 3], [0, 0, 4]]))
# # m = a.getNextPossibleSVNs([[-1, 0, 0], [1, 2, 3], [0, 0, 4]])
# # av = 0
# # for p in m[2]:
# #     print(p.getValue())
# #     av += p.getValue() / 4
# # print("---- " + str(av))
# # for p in m[3]:
# #     print(p.getValue())
# # aendpos = [[7, 6, 1],[8,5,2],[9,4,3]]
# # endpos = [[6, 5, -1],[7,4,1],[8,3,2]]

# # dic = a.deepFlowList.SVNDict
# # print(dic[str(endpos)])
# print(a.decide(pos))
# print(a.getNextPossibleSVNs(pos))
# for i in a.getNextPossibleSVNs(pos):
#     print("====")
#     for j in i:
#         print(j.getState())
#         print(j.getValue())