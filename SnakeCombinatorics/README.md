# Tabular-Q-Learning_In_Snake
This is an experiment with the snake game that finds optimal policies for a given square game board.
I wrote everything from scratch more or less (I used a graphics lib)

--Run displayTest and change the line GAME_SIZE = to whatever boardsize you would like. Boardsizes > 3 may be extremely slow to solve/not
compute because state spaces explode according to the following table (Actual section):

Possible states at a given board length, L:

L | Upper Bound (computed using a simple formula) | Actual

1 : 1 | 1

2 : 236 | 44

3 : 137700 | 2080

4 : 535692272 | 143920

5 : 16475056294100 | 22073136
