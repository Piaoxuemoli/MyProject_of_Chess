#纯特卡洛树搜索算法

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
        self.q = 0  #平均价值
        self.u = 0  #ucb上界
        self.prior_p = prior_p  #动作先验概率

    #根据动作概率分布，扩展节点，动作映射到子节点
    def expand(self, action_priors):
        for action, prob in action_priors:
            if action not in self.children:
                self.children[action] = TreeNode(self, prob)

    #选择子节点，使用置信上界和价值函数
    def select(self, c_puct):
        return max(self.children.items(), key=lambda act_node: act_node[1].get_value(c_puct))

    #获取节点的价值,采用ucb计算u，和直接用价值函数得出的q相加，返回作为节点价值
    def get_value(self, c_puct):
        self.u = (c_puct * self.prior_p * np.sqrt(self.parent.n_visits)) / (1 + self.n_visits)
        return self.q + self.u

    #更新节点的访问次数和价值
    def update(self, leaf_value):
        self.n_visits += 1
        self.q += 1.0*(leaf_value - self.q) / self.n_visits  #更平稳,避免value的突变

    #递归更新节点的访问次数和价值
    def update_recursive(self, leaf_value):
        if self.parent:
            self.parent.update_recursive(-leaf_value)  #更新父节点的价值
        self.update(leaf_value)  #更新自身节点的价值

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

    #搜索树创建
    def simulate(self, state):
        node = self.root
        while not node.is_leaf():
            max_action, node = TreeNode.select(node,self.c_puct)
            state.do_move(max_action)  #调用游戏类的move方法，在棋盘上模拟执行动作
        action_probs, _ = self.policy(state)
        if not state.is_end():
            node.expand(action_probs)  #扩展节点
        leaf_value = self.evaluate_rollout(state)  #用rollout策略评估叶子节点的价值
        node.update_recursive(-leaf_value)  #递归更新父节点的价值

    #推出策略
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
        
    #行动函数：构建搜索树，选择动作，返回动作
    def make_move(self, state):
        for i in range(self.limit):
            state_copy = copy.deepcopy(state)
            self.simulate(state_copy)
        return max(self.root.children.items(), key=lambda act_node: act_node[1].n_visits)[0]
    
    #树更新函数：将上一次行动在子树中检索，若存在则提升为根节点，否则重置搜索树
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
