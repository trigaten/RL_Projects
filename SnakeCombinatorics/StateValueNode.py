import copy
"""StateValueNode (SVN) a type of linked list that stores a Snake board state (2D array), a floating 
point value for the evaluation of the state, and 4 arrays to store SVNs of possible next moves in the
left, right, up, down directions"""
class StateValueNode:
    def __init__(self, state, value, left = [], right = [], up = [], down = []):
        # 2D board state
        self.state = state
        # a floating point value for this 
        self.value = value
        # arrays to store potential next StateValueNodes
        self.left = left
        self.right = right
        self.up = up
        self.down = down
        if left != [] or right != [] or up != [] or down != []:
            self.isEndState = False
        else:
            self.isEndState = True
    def getState(self):
        return self.state

    def getValue(self):
        return self.value

    def setValue(self, value):
        self.value = value

    def getMoves(self):
        return self.left, self.right, self.up, self.down
    
    # methods to add a node to the left, right, up, or down array - if an SVN is added as such this SVN is no longer and end stage
    def addToLeft(self, SVN):
        self.isEndState = False
        self.left.append(SVN)
    
    def addToRight(self, SVN):
        self.isEndState = False
        self.right.append(SVN)

    def addToUp(self, SVN):
        self.isEndState = False
        self.up.append(SVN)
    
    def addToDown(self, SVN):
        self.isEndState = False
        self.down.append(SVN)

    def hasNextStates(self):
        return not self.isEndState
    
    # to String method
    def __str__(self):
        return "State: " + str(self.state) + "\nValue " + str(self.value) + "\nleft: " + str(self.left) + "\nright: " + str(self.right) + "\nup: " + str(self.up) + "\ndown: " + str(self.down)  