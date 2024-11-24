#���ؿ����������㷨

import numpy as np
import copy
from operator import itemgetter

def rollout_policy(state):
    act_probs = np.random.rand(len(state.av))
    return zip(state.av, act_probs)
    
def policy_value_fn(state):
    act_probs=np.ones(len(state.av))/len(state.av)
    return zip(state.av, act_probs),0

class TreeNode:
    def __init__(self, parent, prior_p):
        self.parent = parent
        self.children = {}
        self.n_visits = 0
        self.q = 0  #ƽ����ֵ
        self.u = 0  #ucb�Ͻ�
        self.prior_p = prior_p  #�����������

    #���ݶ������ʷֲ�����չ�ڵ㣬����ӳ�䵽�ӽڵ�
    def expand(self, action_priors):
        for action, prob in action_priors:
            if action not in self.children:
                self.children[action] = TreeNode(self, prob)

    #ѡ���ӽڵ㣬ʹ�������Ͻ�ͼ�ֵ����
    def select(self, c_puct):
        return max(self.children.items(), key=lambda act_node: act_node[1].get_value(c_puct))

    #��ȡ�ڵ�ļ�ֵ,����ucb����u����ֱ���ü�ֵ�����ó���q��ӣ�������Ϊ�ڵ��ֵ
    def get_value(self, c_puct):
        self.u = (c_puct * self.prior_p * np.sqrt(self.parent.n_visits)) / (1 + self.n_visits)
        return self.q + self.u

    #���½ڵ�ķ��ʴ����ͼ�ֵ
    def update(self, leaf_value):
        self.n_visits += 1
        self.q += 1.0*(leaf_value - self.q) / self.n_visits  #��ƽ��,����value��ͻ��

    #�ݹ���½ڵ�ķ��ʴ����ͼ�ֵ
    def update_recursive(self, leaf_value):
        if self.parent:
            self.parent.update_recursive(-leaf_value)  #���¸��ڵ�ļ�ֵ
        self.update(leaf_value)  #��������ڵ�ļ�ֵ

    def is_leaf(self):
        return self.children == {}

    def is_root(self):
        return self.parent is None

class MCTS:
    def __init__(self, policy_value_fn, c_puct=5, limit=5000):
        self.root = TreeNode(None, 1.0)
        self.policy = policy_value_fn
        self.c_puct = c_puct
        self.limit = limit

    #����������
    def simulate(self, state):
        node = self.root
        while not node.is_leaf():
            max_action, node = TreeNode.select(node,self.c_puct)
            state.do_move(max_action)  #������Ϸ���move��������������ģ��ִ�ж���
        action_probs, _ = self.policy(state)
        if not state.is_end():
            node.expand(action_probs)  #��չ�ڵ�
        leaf_value = self.evaluate_rollout(state)  #��rollout��������Ҷ�ӽڵ�ļ�ֵ
        node.update_recursive(-leaf_value)  #�ݹ���¸��ڵ�ļ�ֵ

    #�Ƴ�����
    def evaluate_rollout(self,state):
        player = state.get_current_player()
        for i in range(self.limit//10):
            if state.is_end():
                break
            action_probs= rollout_policy(state)
            max_action = max(action_probs, key=itemgetter(1))[0]
            state.do_move(max_action)
        winner = state.get_winner()
        if winner == player:
            return 1.0
        elif winner == -player:
            return -1.0
        elif winner == 0:
            return 0.0
        
    #�ж�������������������ѡ���������ض���
    def make_move(self, state):
        for i in range(self.limit):
            state_copy = copy.deepcopy(state)
            self.simulate(state_copy)
        return max(self.root.children.items(), key=lambda act_node: act_node[1].n_visits)[0]
    
    #�����º���������һ���ж��������м�����������������Ϊ���ڵ㣬��������������
    def up_date(self, last_move):
        if last_move in self.root.children:
            self.root = self.root.children[last_move]
            self.root.parent = None
        else:
            self.root = TreeNode(None, 1.0)
            
class m_Player:
    def __init__(self, c_puct=5, limit=2000):
        self.mcts = MCTS(policy_value_fn, c_puct, limit)

    def get_action(self, state):
        if len(state.av) > 0:
            move = self.mcts.make_move(state)
            self.mcts.up_date(-1)
            return move
