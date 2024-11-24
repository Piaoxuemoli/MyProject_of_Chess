import paddle
import numpy as np

import paddle.nn as nn 
import paddle.nn.functional as F

config = 8

class Net(paddle.nn.Layer):
    def __init__(self,board_width=config, board_height=config):
        super(Net, self).__init__()
        self.board_width = board_width
        self.board_height = board_height
        # ���������
        self.conv1 = nn.Conv2D(in_channels=4,out_channels=32,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2D(in_channels=32,out_channels=64,kernel_size=3,padding=1)
        self.conv3 = nn.Conv2D(in_channels=64,out_channels=128,kernel_size=3,padding=1)
        # �ж����������
        self.act_conv1 = nn.Conv2D(in_channels=128,out_channels=4,kernel_size=1,padding=0)
        self.act_fc1 = nn.Linear(4*self.board_width*self.board_height,
                                 self.board_width*self.board_height)
        self.val_conv1 = nn.Conv2D(in_channels=128,out_channels=2,kernel_size=1,padding=0)
        self.val_fc1 = nn.Linear(2*self.board_width*self.board_height, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, inputs):
        # ��������� 
        x = F.relu(self.conv1(inputs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # �ж����������
        x_act = F.relu(self.act_conv1(x))
        x_act = paddle.reshape(
                x_act, [-1, 4 * self.board_height * self.board_width])
        
        x_act  = F.log_softmax(self.act_fc1(x_act))        
        # ״̬��ֵ�����
        x_val  = F.relu(self.val_conv1(x))
        x_val = paddle.reshape(
                x_val, [-1, 2 * self.board_height * self.board_width])
        x_val = F.relu(self.val_fc1(x_val))
        x_val = F.tanh(self.val_fc2(x_val))

        return x_act,x_val

class PolicyValueNet():
    def __init__(self, board_width=config, board_height=config,
                 model_file=None, use_gpu=True):
        self.use_gpu = use_gpu
        self.board_width = board_width
        self.board_height = board_height
        self.l2_const = 1e-3  #l2����ϵ��
        #��������
        self.policy_value_net = Net(self.board_width, self.board_height)
        #�����Ż���
        self.optimizer  = paddle.optimizer.Adam(learning_rate=0.02,
                                parameters=self.policy_value_net.parameters(), weight_decay=self.l2_const)
        #����ģ�Ͳ���
        if model_file:
            net_params = paddle.load(model_file)
            self.policy_value_net.set_state_dict(net_params)
            
    def policy_value(self, state_batch):
        #�����ķ��ض������ʺ�״̬��ֵ
        state_batch = paddle.to_tensor(state_batch)
        log_act_probs, value = self.policy_value_net(state_batch)
        act_probs = np.exp(log_act_probs.numpy())
        return act_probs, value.numpy()

    def policy_value_fn(self, board):
        #����״̬�������磬���ض������ʺ�״̬��ֵ
        legal_positions = board.av
        current_state = np.ascontiguousarray(board.current_state().reshape(
                -1, 4, self.board_width, self.board_height)).astype("float32")
        current_state = paddle.to_tensor(current_state)
        log_act_probs, value = self.policy_value_net(current_state)
        act_probs = np.exp(log_act_probs.numpy().flatten())  #չƽ
        act_probs = zip(legal_positions, act_probs[legal_positions])
        return act_probs, value.numpy()

    def train_step(self, state_batch, mcts_probs, winner_batch, lr=0.002):
        state_batch = paddle.to_tensor(state_batch)
        mcts_probs = paddle.to_tensor(mcts_probs)
        winner_batch = paddle.to_tensor(winner_batch)
        #�����ݶ�
        self.optimizer.clear_gradients()
        #����ѧϰ��
        self.optimizer.set_lr(lr)
        #��ǰ����
        log_act_probs, value = self.policy_value_net(state_batch)
        #������ʧ
        value = paddle.reshape(x=value, shape=[-1])
        value_loss = F.mse_loss(input=value, label=winner_batch)
        policy_loss = -paddle.mean(paddle.sum(mcts_probs*log_act_probs, axis=1))
        loss = value_loss + policy_loss
        #���򴫲�
        loss.backward()
        self.optimizer.minimize(loss)
        #�������������
        entropy = -paddle.mean(
                paddle.sum(paddle.exp(log_act_probs) * log_act_probs, axis=1)
                )
        return loss.numpy(), entropy.numpy()    

    def get_policy_param(self):
        net_params = self.policy_value_net.state_dict()
        return net_params

    def save_model(self, model_file):
        net_params = self.get_policy_param()
        paddle.save(net_params, model_file)

