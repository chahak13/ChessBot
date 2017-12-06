#!/usr/bin/env pypy
# -*- coding: utf-8 -*-

from __future__ import print_function
import re, sys, time
import RPi.GPIO as GPIO ## Import GPIO Library.
import time ## Import â€˜timeâ€™ library for a delay.
import time
from firebase import firebase
firebase = firebase.FirebaseApplication('https://chessbot-2873e.firebaseio.com/Users/',None)



from itertools import count
from collections import OrderedDict, namedtuple

path = "/home/pi/Desktop/lastPos.txt"

###############################################################################
# Piece-Square tables. Tune these to change sunfish's behaviour
###############################################################################

piece = { 'P': 100, 'N': 280, 'B': 320, 'R': 479, 'Q': 929, 'K': 60000 }
pst = {
    'P': (   0,   0,   0,   0,   0,   0,   0,   0,
            78,  83,  86,  73, 102,  82,  85,  90,
             7,  29,  21,  44,  40,  31,  44,   7,
           -17,  16,  -2,  15,  14,   0,  15, -13,
           -26,   3,  10,   9,   6,   1,   0, -23,
           -22,   9,   5, -11, -10,  -2,   3, -19,
           -31,   8,  -7, -37, -36, -14,   3, -31,
             0,   0,   0,   0,   0,   0,   0,   0),
    'N': ( -66, -53, -75, -75, -10, -55, -58, -70,
            -3,  -6, 100, -36,   4,  62,  -4, -14,
            10,  67,   1,  74,  73,  27,  62,  -2,
            24,  24,  45,  37,  33,  41,  25,  17,
            -1,   5,  31,  21,  22,  35,   2,   0,
           -18,  10,  13,  22,  18,  15,  11, -14,
           -23, -15,   2,   0,   2,   0, -23, -20,
           -74, -23, -26, -24, -19, -35, -22, -69),
    'B': ( -59, -78, -82, -76, -23,-107, -37, -50,
           -11,  20,  35, -42, -39,  31,   2, -22,
            -9,  39, -32,  41,  52, -10,  28, -14,
            25,  17,  20,  34,  26,  25,  15,  10,
            13,  10,  17,  23,  17,  16,   0,   7,
            14,  25,  24,  15,   8,  25,  20,  15,
            19,  20,  11,   6,   7,   6,  20,  16,
            -7,   2, -15, -12, -14, -15, -10, -10),
    'R': (  35,  29,  33,   4,  37,  33,  56,  50,
            55,  29,  56,  67,  55,  62,  34,  60,
            19,  35,  28,  33,  45,  27,  25,  15,
             0,   5,  16,  13,  18,  -4,  -9,  -6,
           -28, -35, -16, -21, -13, -29, -46, -30,
           -42, -28, -42, -25, -25, -35, -26, -46,
           -53, -38, -31, -26, -29, -43, -44, -53,
           -30, -24, -18,   5,  -2, -18, -31, -32),
    'Q': (   6,   1,  -8,-104,  69,  24,  88,  26,
            14,  32,  60, -10,  20,  76,  57,  24,
            -2,  43,  32,  60,  72,  63,  43,   2,
             1, -16,  22,  17,  25,  20, -13,  -6,
           -14, -15,  -2,  -5,  -1, -10, -20, -22,
           -30,  -6, -13, -11, -16, -11, -16, -27,
           -36, -18,   0, -19, -15, -15, -21, -38,
           -39, -30, -31, -13, -31, -36, -34, -42),
    'K': (   4,  54,  47, -99, -99,  60,  83, -62,
           -32,  10,  55,  56,  56,  55,  10,   3,
           -62,  12, -57,  44, -67,  28,  37, -31,
           -55,  50,  11,  -4, -19,  13,   0, -49,
           -55, -43, -52, -28, -51, -47,  -8, -50,
           -47, -42, -43, -79, -64, -32, -29, -32,
            -4,   3, -14, -50, -57, -18,  13,   4,
            17,  30,  -3, -14,   6,  -1,  40,  18),
}
# Pad tables and join piece and pst dictionaries
for k, table in pst.items():
    padrow = lambda row: (0,) + tuple(x+piece[k] for x in row) + (0,)
    pst[k] = sum((padrow(table[i*8:i*8+8]) for i in range(8)), ())
    pst[k] = (0,)*20 + pst[k] + (0,)*20

###############################################################################
# Global constants
###############################################################################

# Our board is represented as a 120 character string. The padding allows for
# fast detection of moves that don't stay within the board.
A1, H1, A8, H8 = 91, 98, 21, 28
initial = (
    '         \n'  #   0 -  9
    '         \n'  #  10 - 19
    ' rnbqkbnr\n'  #  20 - 29
    ' pppppppp\n'  #  30 - 39
    ' ........\n'  #  40 - 49
    ' ........\n'  #  50 - 59
    ' ........\n'  #  60 - 69
    ' ........\n'  #  70 - 79
    ' PPPPPPPP\n'  #  80 - 89
    ' RNBQKBNR\n'  #  90 - 99
    '         \n'  # 100 -109
    '         \n'  # 110 -119
)

# Lists of possible moves for each piece type.
N, E, S, W = -10, 1, 10, -1
directions = {
    'P': (N, N+N, N+W, N+E),
    'N': (N+N+E, E+N+E, E+S+E, S+S+E, S+S+W, W+S+W, W+N+W, N+N+W),
    'B': (N+E, S+E, S+W, N+W),
    'R': (N, E, S, W),
    'Q': (N, E, S, W, N+E, S+E, S+W, N+W),
    'K': (N, E, S, W, N+E, S+E, S+W, N+W)
}

# Mate value must be greater than 8*queen + 2*(rook+knight+bishop)
# King value is set to twice this value such that if the opponent is
# 8 queens up, but we got the king, we still exceed MATE_VALUE.
# When a MATE is detected, we'll set the score to MATE_UPPER - plies to get there
# E.g. Mate in 3 will be MATE_UPPER - 6
MATE_LOWER = piece['K'] - 10*piece['Q']
MATE_UPPER = piece['K'] + 10*piece['Q']

# The table size is the maximum number of elements in the transposition table.
TABLE_SIZE = 1e8

# Constants for tuning search
QS_LIMIT = 150
EVAL_ROUGHNESS = 20


###############################################################################
# Chess logic
###############################################################################

class Position(namedtuple('Position', 'board score wc bc ep kp')):
    """ A state of a chess game
    board -- a 120 char representation of the board
    score -- the board evaluation
    wc -- the castling rights, [west/queen side, east/king side]
    bc -- the opponent castling rights, [west/king side, east/queen side]
    ep - the en passant square
    kp - the king passant square
    """

    def gen_moves(self):
        # For each of our pieces, iterate through each possible 'ray' of moves,
        # as defined in the 'directions' map. The rays are broken e.g. by
        # captures or immediately in case of pieces such as knights.
        for i, p in enumerate(self.board):
            if not p.isupper(): continue
            for d in directions[p]:
                for j in count(i+d, d):
                    q = self.board[j]
                    # Stay inside the board, and off friendly pieces
                    if q.isspace() or q.isupper(): break
                    # Pawn move, double move and capture
                    if p == 'P' and d in (N, N+N) and q != '.': break
                    if p == 'P' and d == N+N and (i < A1+N or self.board[i+N] != '.'): break
                    if p == 'P' and d in (N+W, N+E) and q == '.' and j not in (self.ep, self.kp): break
                    # Move it
                    yield (i, j)
                    # Stop crawlers from sliding, and sliding after captures
                    if p in 'PNK' or q.islower(): break
                    # Castling, by sliding the rook next to the king
                    if i == A1 and self.board[j+E] == 'K' and self.wc[0]: yield (j+E, j+W)
                    if i == H1 and self.board[j+W] == 'K' and self.wc[1]: yield (j+W, j+E)

    def rotate(self):
        ''' Rotates the board, preserving enpassant '''
        return Position(
            self.board[::-1].swapcase(), -self.score, self.bc, self.wc,
            119-self.ep if self.ep else 0,
            119-self.kp if self.kp else 0)

    def nullmove(self):
        ''' Like rotate, but clears ep and kp '''
        return Position(
            self.board[::-1].swapcase(), -self.score,
            self.bc, self.wc, 0, 0)

    def move(self, move):
        i, j = move
        p, q = self.board[i], self.board[j]
        put = lambda board, i, p: board[:i] + p + board[i+1:]
        # Copy variables and reset ep and kp
        board = self.board
        wc, bc, ep, kp = self.wc, self.bc, 0, 0
        score = self.score + self.value(move)
        # Actual move
        board = put(board, j, board[i])
        board = put(board, i, '.')
        # Castling rights, we move the rook or capture the opponent's
        if i == A1: wc = (False, wc[1])
        if i == H1: wc = (wc[0], False)
        if j == A8: bc = (bc[0], False)
        if j == H8: bc = (False, bc[1])
        # Castling
        if p == 'K':
            wc = (False, False)
            if abs(j-i) == 2:
                kp = (i+j)//2
                board = put(board, A1 if j < i else H1, '.')
                board = put(board, kp, 'R')
        # Pawn promotion, double move and en passant capture
        if p == 'P':
            if A8 <= j <= H8:
                board = put(board, j, 'Q')
            if j - i == 2*N:
                ep = i + N
            if j - i in (N+W, N+E) and q == '.':
                board = put(board, j+S, '.')
        # We rotate the returned position, so it's ready for the next player
        return Position(board, score, wc, bc, ep, kp).rotate()

    def value(self, move):
        i, j = move
        p, q = self.board[i], self.board[j]
        # Actual move
        score = pst[p][j] - pst[p][i]
        # Capture
        if q.islower():
            score += pst[q.upper()][119-j]
        # Castling check detection
        if abs(j-self.kp) < 2:
            score += pst['K'][119-j]
        # Castling
        if p == 'K' and abs(i-j) == 2:
            score += pst['R'][(i+j)//2]
            score -= pst['R'][A1 if j < i else H1]
        # Special pawn stuff
        if p == 'P':
            if A8 <= j <= H8:
                score += pst['Q'][j] - pst['P'][j]
            if j == self.ep:
                score += pst['P'][119-(j+S)]
        return score

###############################################################################
# Search logic
###############################################################################

# lower <= s(pos) <= upper
Entry = namedtuple('Entry', 'lower upper')

# The normal OrderedDict doesn't update the position of a key in the list,
# when the value is changed.
class LRUCache:
    '''Store items in the order the keys were last added'''
    def __init__(self, size):
        self.od = OrderedDict()
        self.size = size

    def get(self, key, default=None):
        try: self.od.move_to_end(key)
        except KeyError: return default
        return self.od[key]

    def __setitem__(self, key, value):
        try: del self.od[key]
        except KeyError:
            if len(self.od) == self.size:
                self.od.popitem(last=False)
        self.od[key] = value

class Searcher:
    def __init__(self):
        self.tp_score = LRUCache(TABLE_SIZE)
        self.tp_move = LRUCache(TABLE_SIZE)
        self.nodes = 0

    def bound(self, pos, gamma, depth, root=True):
        """ returns r where
                s(pos) <= r < gamma    if gamma > s(pos)
                gamma <= r <= s(pos)   if gamma <= s(pos)"""
        self.nodes += 1

        # Depth <= 0 is QSearch. Here any position is searched as deeply as is needed for calmness, and so there is no reason to keep different depths in the transposition table.
        depth = max(depth, 0)

        # Sunfish is a king-capture engine, so we should always check if we
        # still have a king. Notice since this is the only termination check,
        # the remaining code has to be comfortable with being mated, stalemated
        # or able to capture the opponent king.
        if pos.score <= -MATE_LOWER:
            return -MATE_UPPER

        # Look in the table if we have already searched this position before.
        # We also need to be sure, that the stored search was over the same
        # nodes as the current search.
        entry = self.tp_score.get((pos, depth, root), Entry(-MATE_UPPER, MATE_UPPER))
        if entry.lower >= gamma and (not root or self.tp_move.get(pos) is not None):
            return entry.lower
        if entry.upper < gamma:
            return entry.upper

        # Here extensions may be added
        # Such as 'if in_check: depth += 1'

        # Generator of moves to search in order.
        # This allows us to define the moves, but only calculate them if needed.
        def moves():
            # First try not moving at all
            if depth > 0 and not root and any(c in pos.board for c in 'RBNQ'):
                yield None, -self.bound(pos.nullmove(), 1-gamma, depth-3, root=False)
            # For QSearch we have a different kind of null-move
            if depth == 0:
                yield None, pos.score
            # Then killer move. We search it twice, but the tp will fix things for us. Note, we don't have to check for legality, since we've already done it before. Also note that in QS the killer must be a capture, otherwise we will be non deterministic.
            killer = self.tp_move.get(pos)
            if killer and (depth > 0 or pos.value(killer) >= QS_LIMIT):
                yield killer, -self.bound(pos.move(killer), 1-gamma, depth-1, root=False)
            # Then all the other moves
            for move in sorted(pos.gen_moves(), key=pos.value, reverse=True):
                if depth > 0 or pos.value(move) >= QS_LIMIT:
                    yield move, -self.bound(pos.move(move), 1-gamma, depth-1, root=False)

        # Run through the moves, shortcutting when possible
        best = -MATE_UPPER
        for move, score in moves():
            best = max(best, score)
            if best >= gamma:
                # Save the move for pv construction and killer heuristic
                self.tp_move[pos] = move
                break

        # Stalemate checking is a bit tricky: Say we failed low, because
        # we can't (legally) move and so the (real) score is -infty.
        # At the next depth we are allowed to just return r, -infty <= r < gamma,
        # which is normally fine.
        # However, what if gamma = -10 and we don't have any legal moves?
        # Then the score is actaully a draw and we should fail high!
        # Thus, if best < gamma and best < 0 we need to double check what we are doing.
        # This doesn't prevent sunfish from making a move that results in stalemate,
        # but only if depth == 1, so that's probably fair enough.
        # (Btw, at depth 1 we can also mate without realizing.)
        if best < gamma and best < 0 and depth > 0:
            is_dead = lambda pos: any(pos.value(m) >= MATE_LOWER for m in pos.gen_moves())
            if all(is_dead(pos.move(m)) for m in pos.gen_moves()):
                in_check = is_dead(pos.nullmove())
                best = -MATE_UPPER if in_check else 0

        # Table part 2
        if best >= gamma:
            self.tp_score[(pos, depth, root)] = Entry(best, entry.upper)
        if best < gamma:
            self.tp_score[(pos, depth, root)] = Entry(entry.lower, best)

        return best

    # secs over maxn is a breaking change. Can we do this?
    # I guess I could send a pull request to deep pink
    # Why include secs at all?
    def _search(self, pos):
        """ Iterative deepening MTD-bi search """
        self.nodes = 0

        # In finished games, we could potentially go far enough to cause a recursion
        # limit exception. Hence we bound the ply.
        for depth in range(1, 1000):
            self.depth = depth
            # The inner loop is a binary search on the score of the position.
            # Inv: lower <= score <= upper
            # 'while lower != upper' would work, but play tests show a margin of 20 plays better.
            lower, upper = -MATE_UPPER, MATE_UPPER
            while lower < upper - EVAL_ROUGHNESS:
                gamma = (lower+upper+1)//2
                score = self.bound(pos, gamma, depth)
                if score >= gamma:
                    lower = score
                if score < gamma:
                    upper = score
            # We want to make sure the move to play hasn't been kicked out of the table,
            # So we make another call that must always fail high and thus produce a move.
            score = self.bound(pos, lower, depth)

            # Yield so the user may inspect the search
            yield

    def search(self, pos, secs):
        start = time.time()
        for _ in self._search(pos):
            if time.time() - start > secs:
                break
        # If the game hasn't finished we can retrieve our move from the
        # transposition table.
        return self.tp_move.get(pos), self.tp_score.get((pos, self.depth, True)).lower


###############################################################################
# User interface
###############################################################################

# Python 2 compatability
if sys.version_info[0] == 2:
    input = raw_input
    class NewOrderedDict(OrderedDict):
        def move_to_end(self, key):
            value = self.pop(key)
            self[key] = value
    OrderedDict = NewOrderedDict


def parse(c):
    fil, rank = ord(c[0]) - ord('a'), int(c[1]) - 1
    return A1 + fil - 10*rank


def render(i):
    rank, fil = divmod(i - A1, 10)
    return chr(fil + ord('a')) + str(-rank + 1)


"""def print_pos(pos):
    print()
    uni_pieces = {'R':'Ã¢â„¢Å“', 'N':'Ã¢â„¢Å¾', 'B':'Ã¢â„¢Â', 'Q':'Ã¢â„¢â€º', 'K':'Ã¢â„¢Å¡', 'P':'Ã¢â„¢Å¸',
                  'r':'Ã¢â„¢â€“', 'n':'Ã¢â„¢Ëœ', 'b':'Ã¢â„¢â€”', 'q':'Ã¢â„¢â€¢', 'k':'Ã¢â„¢â€', 'p':'Ã¢â„¢â„¢', '.':'Ã‚Â·'}
    for i, row in enumerate(pos.board.split()):
        print(' ', 8-i, ' '.join(uni_pieces.get(p, p) for p in row))
    print('    a b c d e f g h \n\n')
"""
#global oneStep, cur_x, cur_y, angle1, duty1, angle2, duty2, pwm
oneStep = 7
cur_x = 0
cur_y = 0


GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)
GPIO.setup(26, GPIO.OUT) ## set output.

pwm=GPIO.PWM(26,50) ## PWM Frequency
pwm.start(7)

angle1=90
duty1= 7 ## Angle To Duty cycle Conversion

angle2=180
duty2= 11
servoDelay=50.0/1000.0

def moveUp():
    pwm.ChangeDutyCycle(duty2)
    time.sleep(servoDelay)

def moveDown():
    pwm.ChangeDutyCycle(duty1)
    time.sleep(servoDelay)
    #GPIO.cleanup()

# 0 1 x-axis
# 2 3 y-axis

coil_A_1_pin = [3,12,23,18]
coil_A_2_pin = [7,15,11,21]
coil_B_1_pin = [5,13,24,19]
coil_B_2_pin = [11,16,16,22]


# adjust if different
StepCount = 8
Seq = range(0, StepCount)
#
Seq[0] = [1,0,0,0]
Seq[1] = [1,1,0,0]
#
Seq[2] = [0,1,0,0]
Seq[3] = [0,1,1,0]
#
Seq[4] = [0,0,1,0]
Seq[5] = [0,0,1,1]
#
Seq[6] = [0,0,0,1]
Seq[7] = [1,0,0,1]

# 1-> white  2->black 
chessboard = [
                [0,0,0,0,0,0,0,0,0],
                [0,1,1,1,1,1,1,1,1],
                [0,1,1,1,1,1,1,1,1],
                [0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0],
                [0,2,2,2,2,2,2,2,2],
                [0,2,2,2,2,2,2,2,2]
            ]
"""
chessboard[1] =[0,1,1,1,1,1,1,1,1]
chessboard[2] =[0,1,1,1,1,1,1,1,1]
chessboard[7] =[0,2,2,2,2,2,2,2,2]
chessboard[8] =[0,2,2,2,2,2,2,2,2]
"""
 
# GPIO.setup(enable_pin, GPIO.OUT)

for i in range(4):
    GPIO.setup(coil_A_1_pin[i], GPIO.OUT)
    GPIO.setup(coil_A_2_pin[i], GPIO.OUT)
    GPIO.setup(coil_B_1_pin[i], GPIO.OUT)
    GPIO.setup(coil_B_2_pin[i], GPIO.OUT)

# GPIO.output(enable_pin, 1)

def setStep(m1,w1, w2, w3, w4,m2,v1,v2,v3,v4):
    
    GPIO.output(coil_A_1_pin[m1], w1)
    GPIO.output(coil_A_1_pin[m2], v1)
    
    GPIO.output(coil_A_2_pin[m1], w2)
    GPIO.output(coil_A_2_pin[m2], v2)
    
    GPIO.output(coil_B_1_pin[m1], w3)
    GPIO.output(coil_B_1_pin[m2], v3)
    
    GPIO.output(coil_B_2_pin[m1], w4)
    GPIO.output(coil_B_2_pin[m2], v4)

Delay=9

def forward(m1,m2,delay,steps):
    print("move ",m1," ",m2," ",steps)
    if steps >= 0:
        for i in range(steps):
            for j in range(StepCount):
                setStep(m1,Seq[j][0], Seq[j][1], Seq[j][2], Seq[j][3],m2,Seq[7-j][0], Seq[7-j][1], Seq[7-j][2], Seq[7-j][3])
                time.sleep(delay/1000.0)
    else:
        for i in range(-1*steps):
            for j in range(StepCount):
                setStep(m1,Seq[7-j][0], Seq[7-j][1], Seq[7-j][2], Seq[7-j][3],m2, Seq[j][0], Seq[j][1], Seq[j][2], Seq[j][3])
                time.sleep(delay/1000.0)

def moveMotor(x1,y1):
    global cur_x, cur_y
    hor=-cur_x+x1
    ver=-cur_y+y1
    print("hor ",hor," ver ",ver)
    print("move to ",x1," ",y1)
    #move motors to top right corner
    if hor>=0:
        forward(2,3,Delay,oneStep+2*hor*oneStep)
    else :
        forward(2,3,Delay,oneStep+2*hor*oneStep)

    if ver>=0:
        forward(0,1,Delay,oneStep+2*ver*oneStep)
    else :
        forward(0,1,Delay,oneStep+2*ver*oneStep)

    # move motor to center cell
    forward(0,1,Delay,-oneStep)
    forward(2,3,Delay,-oneStep)

    cur_x=x1
    cur_y=y1

def placeMove(player,temp):
    global chessboard

    req_x1=ord(temp[0])-ord('a')+1
    req_y1=ord(temp[1])-ord('0')
    req_x2=ord(temp[2])-ord('a')+1
    req_y2=ord(temp[3])-ord('0')
    
    #print(req_x1,req_y1,req_x2,req_y2)
    #piece already present
    
    if (chessboard[req_y2][req_x2]!=0):

        moveMotor(req_x2,req_y2)
        
        #take piece
        moveUp()

        if chessboard[cur_y][cur_x]==1:
            print("inside 1")
            #move left
            forward(2,3,Delay,-cur_x*oneStep)

            #detach piece
            moveDown()

            #back to cell
            forward(2,3,Delay,cur_x*oneStep)

        elif chessboard[cur_y][cur_x]==2:
            print("inside 2")            
            #move right
            forward(2,3,Delay,(9-cur_x)*oneStep)

            #detach piece
            moveDown()

            #back to cell
            forward(2,3,Delay,-(9-cur_x)*oneStep)

    
    moveMotor(req_x1,req_y1)
    
    chessboard[req_y1][req_x1]=0
    
    #take peice
    moveUp()

    moveMotor(req_x2,req_y2)
    chessboard[req_y2][req_x2]=player
    
    #detach peice
    moveDown()  

def printChess():
    global chessboard
    for i in range(9):
        print(chessboard[i])

def main():
    global cur_x,cur_y

    ######read last position########
    fp = open(path,"r")
    lastPos = fp.read().splitlines()
    cur_x = int(lastPos[0].split(' ')[0])
    cur_y = int(lastPos[0].split(' ')[1])
    fp.close()
    ################################

    pos = Position(initial, 0, (True,True), (True,True), 0, 0)
    searcher = Searcher()
    global chessboard
    
    
    
    while True:
        print('hi')

        if pos.score <= -MATE_LOWER:
            print("You lost")
            break

        # We query the user until she enters a (pseudo) legal move.
        move = None
        temp ="error"
        while move not in pos.gen_moves():

            #print_pos(pos)
            #print 'Connect with ' + addr[0] + ':' + str(addr[1])
            buf = str(firebase.get('move',None))
            temp = buf.replace(" ","").lower()

            match = re.match('([a-h][1-8])'*2, temp)
            if match:
                move = parse(match.group(1)), parse(match.group(2))
            else:
                # Inform the user when invalid input (e.g. "help") is entered
                print("Please enter a move like g8f6")
                time.sleep(3)
        #fp.write("Mmove "+temp+"\n")
        
        print("Your move:", temp)
        pos = pos.move(move)
        placeMove(1,temp)
        print("Move placed")
        printChess()
        
        # After our move we rotate the board and print it again.
        # This allows us to see the effect of our move.

        if pos.score <= -MATE_LOWER:
            print("You won")
            break

        # Fire up the engine to look for a move.
        move, score = searcher.search(pos, secs=2)

        if score == MATE_UPPER:
            print("Checkmate!")

        
        # The black player moves from a rotated position, so we have to
        # 'back rotate' the move before printing it.
        temp=render(119-move[0]) + render(119-move[1])
        print("My move:", render(119-move[0]) + render(119-move[1]))
        placeMove(2,temp)
        
        
        pos = pos.move(move)
        printChess()
    ######store last position########
    fp = open(path,"w")
    fp.write(str(cur_x)+" "+str(cur_y))
    fp.close()
    ################################

if __name__ == '__main__':
    global s
    main()
    s.close()
