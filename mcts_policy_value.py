import numpy as np
import copy

#带网络的mcts搜索树(网络外接)

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

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
        action_probs, leaf_value = self.policy(state) #网络直接预测先验概率，动作，价值
        if not state.is_end():
            node.expand(action_probs)  #不是叶子就扩展节点
        else:
            leaf_value = state.get_winner()  #若游戏结束，胜者作为叶子价值，无需使用估计值
        node.update_recursive(-leaf_value)  #递归更新父节点的价值
        
    #行动函数：构建搜索树，选择动作，返回动作
    def make_move(self, state,temp=1e-3):
        for i in range(self.limit):
            state_copy = copy.deepcopy(state)
            self.simulate(state_copy)
        act_visits = [(act, node.n_visits)
                      for act, node in self.root.children.items()]
        acts, visits = zip(*act_visits)
        #用模拟结束后的访问次数来生成动作（子节点）对应的概率
        probs=softmax(1.0/temp * np.log(np.array(visits) + 1e-10))
        return acts,probs
    
    #树更新函数：将上一次行动在子树中检索，若存在则提升为根节点，否则重置搜索树
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
                #自我训练时添加狄利克雷噪声，以提高探索性
                move = np.random.choice(
                    acts,
                    p=0.85*probs + 0.15*np.random.dirichlet(0.3*np.ones(len(probs)))
                )
                #更新根节点
                self.mcts.up_date(move)
            else:
                move = np.random.choice(acts, p=probs)
                #重置根节点
                self.mcts.up_date(-1)

            if return_prob:
                return move, move_probs
            else:
                return move
        else:
            print("board is full")
