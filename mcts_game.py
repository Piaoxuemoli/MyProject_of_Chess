import numpy as np
import pygame as pg
import sys

#������
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
        # ʹ��4��15x15�Ķ�ֵ����ƽ����������ǰ�ľ���
        # ǰ����ƽ��ֱ��ʾ��ǰplayer������λ�úͶ���player������λ��
        # �����ӵ�λ����1��û���ӵ�λ����0
        # ������ƽ���ʾ����player���һ��������λ��
        # Ҳ��������ƽ��ֻ��һ��λ����1������ȫ����0
        # ���ĸ�ƽ���ʾ���ǵ�ǰplayer�ǲ�������player
        # ���������player������ƽ��ȫ��Ϊ1
        square_state = np.zeros((4, self.width, self.height))
        if self.state:
            moves, players = np.array(list(zip(*self.state.items())))
            move_curr = moves[players == self.current_player]   # ��ȡ��ǰ��ҵ������ƶ�ֵ
            move_oppo = moves[players == -self.current_player]   # ��ȡ�Է���ҵ������ƶ�ֵ
            square_state[0][move_curr // self.width, 
                            move_curr % self.height] = 1.0
            square_state[1][move_oppo // self.width,
                            move_oppo % self.height] = 1.0
            # ָ�����һ���ƶ�λ��
            square_state[2][self.last_move // self.width,
                            self.last_move % self.height] = 1.0
        if len(self.state) % 2 == 0: #����ǰ����������������ƽ��ȫ��Ϊ1
            square_state[3][:, :] = 1.0
        # ��ÿ��ƽ������״̬��������ת��(��һ�л������һ��)
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


# ����UI�Ĳ��ֵ�ѵ����ʽ
class Game_UI(object):
    def __init__(self, state, is_shown, **kwargs):
        self.state = state  
        self.is_shown = is_shown
        #��ʼ��pygame
        pg.init()
        if is_shown != 0:
            self.size = self.height, self.width = config*40+40, config*40+40
            self.__screen = pg.display.set_mode(self.size)
            pg.display.set_caption('Chess of Uzi')

    #�߼�����תΪ��������
    def loc_to_pos(self, x, y):
        return (x+1)*40, (y+1)*40

    #��������תΪ�߼�����
    def pos_to_loc(self, x, y):
        i,j= round(x/40)-1, round(y/40)-1
        #��ֹԽ��
        if i < 0 or i >= config or j < 0 or j >= config:
            return None, None
        else:
            return i, j

    #���̻��ƺ���
    def draw_board(self):
        for i in range(1, config+1):
            pg.draw.line(self.__screen, (0, 0, 0), (40, i*40), (self.width-40, i*40))
            pg.draw.line(self.__screen, (0, 0, 0), (i*40, 40), (i*40, self.height-40)) 

    #���ӻ���        
    def draw_chess(self, screen):
        for move,player in self.state.state.items():
            x,y=move//config,move%config
            if player==1:
                co=(0,0,0)
            else:
                co=(255,255,255)
            pg.draw.circle(screen, co, ((x+1)*40, (y+1)*40), 15)

    def draw(self):
        #���Ʊ���
        pg.draw.rect(self.__screen, (173, 216, 230), (0, 0, self.width, self.height))
        #��������
        self.draw_board()
        #��������
        self.draw_chess(self.__screen)

    def draw_result(self, result):
        font = pg.font.Font(None, 36)
        #���û�ʤ����
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
    
    #ʹ��faker VS bin
    def start_play(self, player1, player2,start_player=1):
        self.board=Board()  # ��ʼ������ 
        self.board.current_player=start_player  # ��������

        if self.is_shown:
            self.draw()
            pg.display.update()

        while True:
            if self.is_shown:
                # ��׽pygame�¼�
                for event in pg.event.get():
                    # �˳�����
                    if event.type == pg.QUIT:
                        pg.quit()
                        exit()

            current_player = self.board.current_player  # ��ȡ��ǰ���
            if current_player == 1:
                player_in_turn = player1  #��ǰ���Ϊ����
            else:
                player_in_turn = player2  #��ǰ���Ϊ����
            move = player_in_turn.get_action(self.board)  # ����MCTS��AI��һ������
            self.board.do_move(move)  # ������һ�����ӵ�״̬�������̸�����
            if self.is_shown:
                self.draw()
                pg.display.update()

            # �жϸþ���Ϸ�Ƿ���ֹ
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
        #�Լ�����,�Ѽ�ѵ������
        self.state=Board()
        #״̬��,mcts����Ϊ����,��ǰ���
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

            # ���ݵ�ǰ����״̬���ؿ��ܵ���Ϊ,����Ϊ��Ӧ�ĸ���
            move, move_probs = player.get_action(self.state,
                                                 temp=temp,
                                                 return_prob=1)
            # �洢����
            states.append(self.state.current_state())  #�洢״̬����
            mcts_probs.append(move_probs)  # �洢��Ϊ��������
            current_players.append(self.state.current_player)  #�洢��ǰ���
            # ִ��һ���ƶ�
            self.state.do_move(move)
            if self.is_shown:
                self.draw()
                pg.display.update()

            # �жϸþ���Ϸ�Ƿ���ֹ
            end=self.state.is_end()
            winner=self.state.get_winner()
            if end:
                #��ÿ��״̬�ĵ�ʱ����ҵĽǶȿ���Ӯ��
                winners_z = np.zeros(len(current_players))
                if winner != 0:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                # ����MSCT�ĸ��ڵ�
                player.reset_player()
                if self.is_shown:
                    self.draw_result(winner)

                    # ˢ��
                    pg.display.update()
                return winner, zip(states, mcts_probs, winners_z)