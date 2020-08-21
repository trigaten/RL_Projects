import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import random
from random import randint

class FillerEnv(gym.Env):  
    metadata = {'render.modes': ['human']}   
    """Contructor takes rows, cols of square board and number of colors to be used in the game
    Also takes extra optional parameter force. This tells the gym whether or not to require 
    that no two adjacent squares have the same color at board setup. adjacencies=False means
    no adjacent squares will have the same color
    """
    def __init__(self, rows=7, columns=8, colors=6, adjacencies=False, opponent="random"):
        if adjacencies == False:
            assert colors > 4, "passing a # < 5 colors lets the same color be in adjacent squares. \nPlease pass a # > 4 or pass \"adjacencies=True\" to allow adjacencies"
        self.adjacencies = adjacencies
        self.rows = rows
        self.columns = columns
        self.colors = colors
        self.board = np.zeros([rows, columns])
        self.p1Squares = []
        self.p2Squares = []
        self.setup()
        # first player's turn
        self.turn = 0
        # stores current colors of player 0 (first player), and player 1
        self.currentColors = [-1, -1]
        
    def step(self, color):
        assert action not in self.currentColors, "that is an illegal action"
        if self.turn = 0:
            self.currentColors[0] = color

        else:
            pass


    def reset(self):
        setBoard()
        self.turn = 0
        self.currentColors = [-1, -1]

    def render(self, mode='human', close=False):
        pass

    def setup(self):
        if self.adjacencies == False:
            # sets up board randomly except that no two adjecent squares may have the same color
            for i, r in enumerate(self.board):
                for j, c in enumerate(r):
                    possibilities = list(range(self.colors))
                    # removes color of square above
                    if i > 0:
                        possibilities.remove(self.board[i-1][j])
                    # removes color of square to left
                    if j > 0:
                        # could be trying to remove a number already removed-- fails silently
                        try:
                            possibilities.remove(self.board[i][j-1])
                        except:
                            pass
                    self.board[i][j] = random.choice(possibilities)
                    print("ff",self.board[i][j])
            print(self.board)
        else:
            # sets up board with completely random colors
            for i, r in enumerate(self.board):
                for j, c in enumerate(r):
                    self.board[i, j] = randint(0, colors-1)

    def actionPossible(action):
        pass
    
    
t = FillerEnv()
