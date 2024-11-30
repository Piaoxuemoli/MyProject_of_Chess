import numpy as np
import copy

#�������mcts������(�������)

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

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
        action_probs, leaf_value = self.policy(state) #����ֱ��Ԥ��������ʣ���������ֵ
        if not state.is_end():
            node.expand(action_probs)  #����Ҷ�Ӿ���չ�ڵ�
        else:
            leaf_value = state.get_winner()  #����Ϸ������ʤ����ΪҶ�Ӽ�ֵ������ʹ�ù���ֵ
        node.update_recursive(-leaf_value)  #�ݹ���¸��ڵ�ļ�ֵ
        
    #�ж�������������������ѡ���������ض���
    def make_move(self, state,temp=1e-3):
        for i in range(self.limit):
            state_copy = copy.deepcopy(state)
            self.simulate(state_copy)
        act_visits = [(act, node.n_visits)
                      for act, node in self.root.children.items()]
        acts, visits = zip(*act_visits)
        #��ģ�������ķ��ʴ��������ɶ������ӽڵ㣩��Ӧ�ĸ���
        probs=softmax(1.0/temp * np.log(np.array(visits) + 1e-10))
        return acts,probs
    
    #�����º���������һ���ж��������м�����������������Ϊ���ڵ㣬��������������
    def up_date(self, last_move):
        if last_move in self.root.children:
            self.root = self.root.children[last_move]
            self.root.parent = None
        else:
            self.root = TreeNode(None, 1.0)
            
class network_MCTS_Player:
    def __init__(self, policy_value_fn ,c_puct=5, limit=1000 ,is_selfplay=0):
        self.mcts = MCTS(policy_value_fn, c_puct, limit)
        self.is_selfplay = is_selfplay

    def set_player(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.up_date(-1)

    def get_action(self, state, temp=1e-3, return_prob=0):
        move_probs = np.zeros(state.config**2)
        if len(state.av) > 0:
            acts, probs = self.mcts.make_move(state, temp)
            move_probs[list(acts)] = probs
            if self.is_selfplay:
                #����ѵ��ʱ��ӵ������������������̽����
                move = np.random.choice(
                    acts,
                    p=0.85*probs + 0.15*np.random.dirichlet(0.3*np.ones(len(probs)))
                )
                #���¸��ڵ�
                self.mcts.up_date(move)
            else:
                move = np.random.choice(acts, p=probs)
                #���ø��ڵ�
                self.mcts.up_date(-1)

            if return_prob:
                return move, move_probs
            else:
                return move
        else:
            print("board is full")
