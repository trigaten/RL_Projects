import arcade
import Combinatorics as comb
import random
import Agent
import math
import numpy as np
# from AppKit import NSScreen #library that gets screen width and height
# print(NSScreen.mainScreen().frame())

# general set up

SCREEN_WIDTH = 1440#NSScreen.mainScreen().frame().size.width
SCREEN_HEIGHT = 900#NSScreen.mainScreen().frame().size.height
SCREEN_TITLE = "Snake Game"
# setting up board stuff
SQUARE_SIZE = 20

# -------------------
# CHANGE THIS INFO
GAME_SIZE = 4
FOOD_COLOR = (0, 120, 200)
# -------------------

rows, cols = (5, 5) 
boardArray = [[comb.makeBoard(GAME_SIZE) for i in range(cols)] for j in range(rows)] 

for x in range(len(boardArray)):
    for y in range(len(boardArray[0])): 
        board = comb.makeBoard(GAME_SIZE)
        # puts head at random location and food opposite that
        Xrand = random.randrange(GAME_SIZE)
        Yrand = random.randrange(GAME_SIZE)
        board[Xrand][Yrand] = 1
        found = False

        while not found:
            X2rand = random.randrange(GAME_SIZE)
            Y2rand = random.randrange(GAME_SIZE)
            if X2rand != Xrand or Y2rand != Yrand:
                board[X2rand][Y2rand] = -1
                found = True
        print(board)
        boardArray[x][y] = board

Still_Going = True

# setup agent
a = Agent.Agent(GAME_SIZE, 0.9, "saves/DFL4")

def on_draw(delta_time):
    """
    Use this function to draw everything to the screen.
    """
    print("fff")
    global Still_Going
    print("Still?")
    
    if Still_Going:
        print(Still_Going)
        # print(board)
        # print(a.decide(board))
        
        # Start the render. This must happen before any drawing
        # commands. We do NOT need a stop render command.
        
        arcade.start_render()
        for x in range(len(boardArray)):
            for y in range(len(boardArray[0])): 
                board = boardArray[x][y]
                next = a.decide(board)
                print("next " + str(next))
                if next == -1:
                    Still_Going = True
                else:
                    head = a.getHeadLocation(board)
                    willEat = False
                    # left
                    if next == 0:
                        if board[head.x][head.y-1] == -1:
                            willEat = True
                        else:
                            willEat = False
                    # right
                    elif next == 1:
                        if board[head.x][head.y+1] == -1:
                            willEat = True
                        else:
                            willEat = False
                    # up
                    elif next == 2:
                        if board[head.x-1][head.y] == -1:
                            willEat = True
                        else:
                            willEat = False
                    # down
                    elif next == 3:
                        if board[head.x+1][head.y] == -1:
                            willEat = True
                        else:
                            willEat = False
                    print("eat?")

                    print(willEat)
                    length = a.getLength(board)
                    for x in range(len(board)):
                            for y in range(len(board[0])):
                                if willEat:
                                    if board[x][y] == -1:
                                        board[x][y] = 1
                                    elif board[x][y] > 0:
                                        board[x][y] += 1
                                else:
                                    if board[x][y] == length:
                                        board[x][y] = 0
                                    elif board[x][y] > 0:
                                        board[x][y] += 1
                                    if next == 0:
                                        board[head.x][head.y-1] = 1
                                    elif next == 1:
                                        board[head.x][head.y+1] = 1
                                    elif next == 2:
                                        board[head.x-1][head.y] = 1
                                    elif next == 3:
                                        board[head.x+1][head.y] = 1
                    
                    if willEat and a.getLength(board) < math.pow(len(board), 2):
                        Xrand = random.randrange(len(board))
                        Yrand = random.randrange(len(board))
                        while board[Xrand][Yrand] > 0:
                            Xrand = random.randrange(GAME_SIZE)
                            Yrand = random.randrange(GAME_SIZE)
                        board[Xrand][Yrand] = -1
            if a.getLength(board) == math.pow(len(board), 2):
                Still_Going = True
            print(board)
        drawBoardArray(boardArray)
    
def drawBoardArray(boards):
    xboards = len(boards)
    yBoards = len(boards[0])
    for x in range(len(boards)):
        for y in range(len(boards[0])): 
            drawBoard(boards[x][y], 300+ x * SQUARE_SIZE*4, 300+y * SQUARE_SIZE*4)



def drawBoard(board, x, y):
    # xSpace = (SCREEN_WIDTH - SQUARE_SIZE * len(board))/2 + SQUARE_SIZE/2
    # ySpace = (SCREEN_HEIGHT - SQUARE_SIZE * len(board[0]))/2 + SQUARE_SIZE/2
    red = 1
    green = 4
    blue = 9
    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] == -1:
                arcade.draw_rectangle_filled(x + i * SQUARE_SIZE, y + j * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE, FOOD_COLOR)
            elif board[i][j] == 0:
                arcade.draw_rectangle_filled(x + i * SQUARE_SIZE, y + j * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE, (255, 0, 0))
            else:
                arcade.draw_rectangle_filled(x + i * SQUARE_SIZE, y + j * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE, (red * (board[i][j]+1), blue * (board[i][j]+1), green * (board[i][j]+1)))
                # if i-1 > 0 and board[i-1][j] > 0:
                #     arcade.draw_line(i * 100 + 50, j * 100 + 50, (i-1) * 100 + 50, j * 100 + 50)
                                
                
            arcade.draw_rectangle_outline(x + i * SQUARE_SIZE, y + j * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE, (255, 0, 0))
            
def main():
    # Open up our window
    arcade.open_window(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
    arcade.set_background_color(arcade.color.BLACK_LEATHER_JACKET)

    # Tell the computer to call the draw command at the specified interval.
    arcade.schedule(on_draw, 1/10)

    # Run the program
    arcade.run()

if __name__ == "__main__":
    main()