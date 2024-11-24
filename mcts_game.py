import numpy as np
import pygame as pg
import sys

#棋盘类
class Board:
    def __init__(self):
        self.config = 8
        self.state={}
        self.winner=0
        self.av=list(range(self.config*self.config))
        self.current_player=1
        self.width=self.config
        self.height=self.config
    
    def current_state(self):
        # 使用4个15x15的二值特征平面来描述当前的局面
        # 前两个平面分别表示当前player的棋子位置和对手player的棋子位置
        # 有棋子的位置是1，没棋子的位置是0
        # 第三个平面表示对手player最近一步的落子位置
        # 也就是整个平面只有一个位置是1，其余全部是0
        # 第四个平面表示的是当前player是不是先手player
        # 如果是先手player则整个平面全部为1
        square_state = np.zeros((4, self.width, self.height))
        if self.state:
            moves, players = np.array(list(zip(*self.state.items())))
            move_curr = moves[players == self.current_player]   # 获取当前玩家的所有移动值
            move_oppo = moves[players == -self.current_player]   # 获取对方玩家的所有移动值
            square_state[0][move_curr // self.width, 
                            move_curr % self.height] = 1.0
            square_state[1][move_oppo // self.width,
                            move_oppo % self.height] = 1.0
            # 指出最后一个移动位置
            square_state[2][self.last_move // self.width,
                            self.last_move % self.height] = 1.0
        if len(self.state) % 2 == 0: #若当前棋手是先手则整个平面全部为1
            square_state[3][:, :] = 1.0
        # 将每个平面棋盘状态按行逆序转换(第一行换到最后一行)
        return square_state[:, ::-1, :]

    def win_check(self):
        board=np.zeros((self.config,self.config))
        for move,player in self.state.items():
            x,y=move//self.config,move%self.config
            board[x][y]=player
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)] 
        for row in range(self.config):
            for col in range(self.config):
                if board[row][col] in [-1, 1]:
                    player = board[row][col]
                    for dr, dc in directions:
                        count = 1
                        for i in range(1, 5):
                            r, c = row + dr * i, col + dc * i
                            if 0 <= r < self.config and 0 <= c < self.config and board[r][c] == player:
                                count += 1
                            else:
                                break
                        if count == 5:
                            self.winner = player
                            return True
        return False

    def is_end(self):
        return len(self.av) == 0 or self.win_check()
  
    def get_current_player(self):
        return self.current_player

    def get_winner(self):
        nothing=self.win_check()
        if len(self.av)==0:
            self.winner=0
        return self.winner
    
    def do_move(self,move):
        if move in self.av:
            self.state[move]=self.current_player
            self.av.remove(move)
            self.current_player=-self.current_player
            self.last_move=move
        
config=8


# 加上UI的布局的训练方式
class Game_UI(object):
    def __init__(self, state, is_shown, **kwargs):
        self.state = state  
        self.is_shown = is_shown
        #初始化pygame
        pg.init()
        if is_shown != 0:
            self.size = self.height, self.width = config*40+40, config*40+40
            self.__screen = pg.display.set_mode(self.size)
            pg.display.set_caption('Chess of Uzi')

    #逻辑坐标转为像素坐标
    def loc_to_pos(self, x, y):
        return (x+1)*40, (y+1)*40

    #棋盘坐标转为逻辑坐标
    def pos_to_loc(self, x, y):
        i,j= round(x/40)-1, round(y/40)-1
        #防止越界
        if i < 0 or i >= config or j < 0 or j >= config:
            return None, None
        else:
            return i, j

    #棋盘绘制函数
    def draw_board(self):
        for i in range(1, config+1):
            pg.draw.line(self.__screen, (0, 0, 0), (40, i*40), (self.width-40, i*40))
            pg.draw.line(self.__screen, (0, 0, 0), (i*40, 40), (i*40, self.height-40)) 

    #棋子绘制        
    def draw_chess(self, screen):
        for move,player in self.state.state.items():
            x,y=move//config,move%config
            if player==1:
                co=(0,0,0)
            else:
                co=(255,255,255)
            pg.draw.circle(screen, co, ((x+1)*40, (y+1)*40), 15)

    def draw(self):
        #绘制背景
        pg.draw.rect(self.__screen, (173, 216, 230), (0, 0, self.width, self.height))
        #绘制棋盘
        self.draw_board()
        #绘制棋子
        self.draw_chess(self.__screen)

    def draw_result(self, result):
        font = pg.font.Font(None, 36)
        #设置获胜画面
        tb=font.render("black win!", True, (255, 0, 0))
        tb_rect=tb.get_rect()
        tb_rect.center=(self.width/2,self.height/2)
        tw=font.render("white win!", True, (255, 0, 0))
        tw_rect=tw.get_rect()
        tw_rect.center=(self.width/2,self.height/2+30)
        tn=font.render("nobody win!", True, (255, 0, 0))
        tn_rect=tw.get_rect()
        tn_rect.center=(self.width/2,self.height/2+30)
        if result==1:
            self.__screen.blit(tb,tb_rect)
        elif result==-1:
            self.__screen.blit(tw,tw_rect)
        else:
            self.__screen.blit(tn,tn_rect)
    
    #使用faker VS bin
    def start_play(self, player1, player2,start_player=1):
        self.board=Board()  # 初始化棋盘 
        self.board.current_player=start_player  # 设置先手

        if self.is_shown:
            self.draw()
            pg.display.update()

        while True:
            if self.is_shown:
                # 捕捉pygame事件
                for event in pg.event.get():
                    # 退出程序
                    if event.type == pg.QUIT:
                        pg.quit()
                        exit()

            current_player = self.board.current_player  # 获取当前玩家
            if current_player == 1:
                player_in_turn = player1  #当前玩家为黑棋
            else:
                player_in_turn = player2  #当前玩家为白棋
            move = player_in_turn.get_action(self.board)  # 基于MCTS的AI下一步落子
            self.board.do_move(move)  # 根据下一步落子的状态更新棋盘各参数
            if self.is_shown:
                self.draw()
                pg.display.update()

            # 判断该局游戏是否终止
            end=self.state.is_end()
            winner=self.state.get_winner()
            if end:
                win = winner
                break
        if self.is_shown:
            self.draw_result(win)
            pg.display.update()
        return win
   
    def start_self_play(self, player, temp=1e-3):
        #自己对弈,搜集训练数据
        self.state=Board()
        #状态池,mcts的行为概率,当前玩家
        states, mcts_probs, current_players = [], [], []
        if self.is_shown:
            self.draw()
            pg.display.update()

        while True:
            if self.is_shown:
                for event in pg.event.get():
                    if event.type == pg.QUIT:
                        pg.quit()
                        exit()

            # 根据当前棋盘状态返回可能得行为,及行为对应的概率
            move, move_probs = player.get_action(self.state,
                                                 temp=temp,
                                                 return_prob=1)
            # 存储数据
            states.append(self.state.current_state())  #存储状态数据
            mcts_probs.append(move_probs)  # 存储行为概率数据
            current_players.append(self.state.current_player)  #存储当前玩家
            # 执行一个移动
            self.state.do_move(move)
            if self.is_shown:
                self.draw()
                pg.display.update()

            # 判断该局游戏是否终止
            end=self.state.is_end()
            winner=self.state.get_winner()
            if end:
                #从每个状态的当时的玩家的角度看待赢家
                winners_z = np.zeros(len(current_players))
                if winner != 0:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                # 重置MSCT的根节点
                player.reset_player()
                if self.is_shown:
                    self.draw_result(winner)

                    # 刷新
                    pg.display.update()
                return winner, zip(states, mcts_probs, winners_z)