from __future__ import print_function
from mcts_game import Board, Game_UI
from mcts_mine import m_Player,MCTS
from mcts_policy_value import network_MCTS_Player
from network import PolicyValueNet
import paddle
import pygame as pg
import numpy as np
import os
import sys
import time

config = 8
def run():
    model_file = 'dist/current_policy.model'  #模型文件路径
    try:
        board = Board()  # 初始化棋盘
        game = Game_UI(board, is_shown=1)  # 创建游戏对象
        model = PolicyValueNet(model_file = model_file)
        ai_player = network_MCTS_Player(model.policy_value_fn, c_puct=5, limit=400)

        game.draw()
        pg.display.update()
        flag = False
        win = None
        while True:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                    exit()
                elif event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                    x, y = round(event.pos[0]/40)-1, round(event.pos[1]/40)-1
                    if x<0 or x>config or y<0 or y>config:  #检测是否在棋盘内
                        continue
                    index=x*config+y
                    if index not in board.av:  #检测是否已经落子
                        continue
                    board.do_move(index)
                    game.draw()
                    pg.display.update()
                    end=board.is_end()
                    winner=board.get_winner()
                    if end:
                        flag = True
                        win = winner
                        break
                    move = ai_player.get_action(board)  # 基于MCTS的AI下一步落子
                    board.do_move(move)  # 根据下一步落子的状态更新棋盘各参数
                    game.draw()
                    pg.display.update()
                    end=board.is_end()
                    winner=board.get_winner()
                    if end:
                        flag = True
                        win = winner
                        break
            if flag:
                game.draw_result(win)
                pg.display.update()
                break
 
    except KeyboardInterrupt:
        print('\n Game wrong')


if __name__ == '__main__':
    device = 'cpu'
    paddle.set_device(device)
    run()
