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
SQUARE_SIZE = 100

# -------------------
# CHANGE THIS INFO
GAME_SIZE = 4
FOOD_COLOR = (0, 120, 200)
# -------------------

rows, cols = (5, 5) 
board = [[0, 1, 2, 3],[7, 6, 5, 4], [8,0,14,13],[9,10,11,12]]


Still_Going = True

def on_draw(delta_time):
    """
    Use this function to draw everything to the screen.
    """
    global Still_Going
    print("Still?")
    
    if Still_Going:
        print(Still_Going)
        # print(board)
        # print(a.decide(board))
        
        # Start the render. This must happen before any drawing
        # commands. We do NOT need a stop render command.
        
        arcade.start_render()
        drawBoard(board, 500, 500)
    
def drawBoardArray(boards):
    xboards = len(boards)
    yBoards = len(boards[0])
    for x in range(len(boards)):
        for y in range(len(boards[0])): 
            drawBoard(boards[x][y], 300+ x * SQUARE_SIZE*4, 300+y * SQUARE_SIZE*4)



def drawBoard(board, x, y):
    # xSpace = (SCREEN_WIDTH - SQUARE_SIZE * len(board))/2 + SQUARE_SIZE/2
    # ySpace = (SCREEN_HEIGHT - SQUARE_SIZE * len(board[0]))/2 + SQUARE_SIZE/2
    red = 0
    green = 8
    blue = 25
    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] == -1:
                arcade.draw_rectangle_filled(x + i * SQUARE_SIZE, y + j * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE, FOOD_COLOR)
            elif board[i][j] == 0:
                arcade.draw_rectangle_filled(x + i * SQUARE_SIZE, y + j * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE, (255, 0, 0))
            else:
                arcade.draw_rectangle_filled(x + i * SQUARE_SIZE, y + j * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE, (red + int(1.5 * (board[i][j]+1)), blue + 4 * (board[i][j]+1), green + 0 * (board[i][j]+1)))
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