from __future__ import print_function
import random
import numpy as np
import os
from collections import defaultdict, deque
from mcts_game import Board, Game_UI
from mcts_mine import m_Player,MCTS
from mcts_policy_value import network_MCTS_Player
from network import PolicyValueNet
import paddle
import csv

config=8

def append_loss_to_csv(epoch, loss):
    with open('losses.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        # 检查文件是否为空，如果为空则写入表头
        is_empty = file.tell() == 0
        if is_empty:
            writer.writerow(['epoch', 'loss'])
        # 写入新的数据行
        writer.writerow([epoch + 1, loss])

class Train():
    def __init__(self, init_model=None, is_shown = 0):
        # 五子棋逻辑和棋盘UI的参数
        self.board_width = config
        self.board_height = config  
        self.board = Board()
        self.is_shown = is_shown
        self.game = Game_UI(self.board, is_shown)
        # 训练参数
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0  #基于KL散度自适应地调整学习率
        self.temp = 0.8  #探索率温度控制
        self.n_playout = 600  #每次移动的模拟次数
        self.c_puct = 5
        self.buffer_size = 12000  #经验池
        self.batch_size = 512  #训练的mini-batch大小
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 5  #每次更新的train_steps数量
        self.kl_targ = 0.02
        self.check_freq = 400  #评估模型的频率
        self.game_batch_num = 800  #训练总轮次
        self.best_win_ratio = 0.0
        self.fake_mcts_limit= 3000  #纯粹的mcts的模拟，用作评估训练策略的对手
        if init_model:
            #从初始的策略价值网开始训练
            self.policy_value_net = PolicyValueNet(self.board_width,
                                                   self.board_height,
                                                   model_file=init_model)
        else:
            #从新的策略价值网络开始训练
            self.policy_value_net = PolicyValueNet(self.board_width,
                                                   self.board_height)
        #定义训练机器人
        self.mcts_player = network_MCTS_Player(self.policy_value_net.policy_value_fn,
                                      c_puct=self.c_puct,
                                      limit=self.n_playout,
                                      is_selfplay=1)

    def get_equi_data(self, play_data):
        #翻转旋转增加数据,五子棋具有旋转镜像不变性
        extend_data = []
        for state, mcts_porb, winner in play_data:
            for i in [1, 2, 3, 4]:
                #逆时针旋转
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(
                    mcts_porb.reshape(self.board_height, self.board_width)), i)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
                #水平翻转
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
        return extend_data

    def collect_selfplay_data(self, n_games=1):
        for i in range(n_games):
            winner, play_data = self.game.start_self_play(self.mcts_player, temp=self.temp)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            #增加数据到经验池，采用双端队列，水满则溢
            play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data)

    def policy_update(self):
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        state_batch= np.array( state_batch).astype("float32")
        
        mcts_probs_batch = [data[1] for data in mini_batch]
        mcts_probs_batch= np.array( mcts_probs_batch).astype("float32")
        
        winner_batch = [data[2] for data in mini_batch]
        winner_batch= np.array( winner_batch).astype("float32")
        
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                state_batch,
                mcts_probs_batch,
                winner_batch,
                self.learn_rate * self.lr_multiplier)
            
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (
                np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                                axis=1)
                         )
            if kl > self.kl_targ * 4: 
                break
        # 自适应调节学习率
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        explained_var_old = (1 -
                             np.var(np.array(winner_batch) - old_v.flatten()) /
                             np.var(np.array(winner_batch)))
        explained_var_new = (1 -
                             np.var(np.array(winner_batch) - new_v.flatten()) /
                             np.var(np.array(winner_batch)))
        print(("kl:{:.5f},"
               "lr_multiplier:{:.3f},"
               "explained_var_old:{:.3f},"
               "explained_var_new:{:.3f}"
               ).format(kl,
                        self.lr_multiplier,
                        explained_var_old,
                        explained_var_new))
        return loss, entropy

    def policy_evaluate(self, n_games=10):
        #与纯蒙特卡洛对抗来监控网络的性能
        current_mcts_player = network_MCTS_Player(self.policy_value_net.policy_value_fn,
                                         c_puct=self.c_puct,
                                         limit=self.n_playout)
        fake_mcts_player = m_Player(c_puct=5,
                                     limit=self.fake_mcts_limit)
        win_cnt = defaultdict(int)
        for i in range(n_games):
            if i % 2 == 0:
                start_player = 1
            else:
                start_player = -1
            winner = self.game.start_play(current_mcts_player,
                                          fake_mcts_player,
                                          start_player)
            win_cnt[winner] += 1
        win_ratio = 1.0 * (win_cnt[1] + 0.5 * win_cnt[0]) / n_games
        print("limit:{}, win: {}, lose: {}, tie:{}".format(
            self.fake_mcts_limit,
            win_cnt[1], win_cnt[-1], win_cnt[0]))
        return win_ratio

    def run(self):
        root = os.getcwd()

        dst_path = os.path.join(root, 'dist')

        if not os.path.exists(dst_path):
            os.makedirs(dst_path)

        try:
            for i in range(self.game_batch_num):
                self.collect_selfplay_data(self.play_batch_size)
                print("batch i:{}, episode_len:{}".format(
                    i + 1, self.episode_len))
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_update()
                    append_loss_to_csv(i, loss)
                    print("loss :{}, entropy:{}".format(loss, entropy))
                if (i + 1) % 50 == 0:
                    self.policy_value_net.save_model(os.path.join(dst_path, 'current_policy_step.model'))
                # 检查当前模型的性能，保存模型的参数
                if (i + 1) % self.check_freq == 0:
                    print("current self-play batch: {}".format(i + 1))
                    win_ratio = self.policy_evaluate()
                    self.policy_value_net.save_model(os.path.join(dst_path, 'current_policy.model'))
                    if win_ratio > self.best_win_ratio:
                        print("faker666")
                        self.best_win_ratio = win_ratio
                        # 更新最好的策略
                        self.policy_value_net.save_model(os.path.join(dst_path, 'best_policy.model'))
                        if (self.best_win_ratio == 1.0 and self.fake_mcts_limit < 8000):
                            self.fake_mcts_limit += 1000
                            self.best_win_ratio = 0.0
        except KeyboardInterrupt:
            print('\n\rquit')


if __name__ == '__main__':
        device = 'gpu:0'
        paddle.set_device(device)
        is_shown = 0
        model_path = 'dist/current_policy_step.model'
        training = Train(model_path,is_shown)
        training.run()

