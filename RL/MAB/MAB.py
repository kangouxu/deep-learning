#拉杆式10臂老虎机
#中奖 1  不中 0
#伯努利分布的概率
import numpy as np
import matplotlib.pyplot as plt

class BernoulliBandit:
    def __init__(self,k):
        self.probs =np.random.uniform(size=k)#随机生成k个数，作为中奖概率
        self.best_id= np.argmax(self.probs)# best one
        self.best_prob = self.probs[self.best_id]
        self.k=k
    def step(self,k):
        if np.random.rand() < self.probs[k]:
            return 1
        else:
            return 0
np.random.seed(1)
k = 10 # arm_num
bandit_10_arm =BernoulliBandit(k)
print("create a bernoullibandit with %d bars"%k)
print("the best bar is %d ,and chance is %.4f"%(bandit_10_arm.best_id,bandit_10_arm.best_prob))

class Solver:
    def __init__(self,bandit):
        self.bandit =bandit
        self.counts = np.zeros(self.bandit.k)
        self.regret = 0 # regret
        self.action =[]# every step's action
        self.regrets = []#regret list
    def update_regret(self,k):
        self.regret = self.regret + (self.bandit.best_prob - self.bandit.probs[k])
        self.regrets.append(self.regret)
    def run_one_step(self):  #one step policy
        raise NotImplementedError
    def run(self,step_nums):
        for _ in range(step_nums):
            k =self.run_one_step()
            self.counts[k]+=1
            self.action.append(k)
            self.update_regret(k)

class EpsilonGreedy(Solver): 
    """ epsilon 贪婪算法,继承 Solver 类 """ 
    def __init__(self, bandit, epsilon=0.01, init_prob=1.0):
        super(EpsilonGreedy, self).__init__(bandit) 
        self.epsilon = epsilon 
        #初始化拉动所有拉杆的期望奖励估值
        self.estimates = np.array([init_prob] * self.bandit.k) 
    def run_one_step(self): 
        if np.random.random() < self.epsilon: 
            k = np.random.randint(0, self.bandit.k) # 随机选择一根拉杆
        else: 
            k = np.argmax(self.estimates) # 选择期望奖励估值最大的拉杆
        r = self.bandit.step(k) # 得到本次动作的奖励
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k]) 
        return k

def plot_results(solvers, solver_names): 
    for idx, solver in enumerate(solvers): 
        time_list = range(len(solver.regrets)) 
        plt.plot(time_list, solver.regrets, label=solver_names[idx]) 
        plt.xlabel('Time steps') 
        plt.ylabel('Cumulative regrets') 
        plt.title('%d-armed bandit' % solvers[0].bandit.k) 
        plt.legend() 
    plt.show()

np.random.seed(1) 
epsilon_greedy_solver = EpsilonGreedy(bandit_10_arm, epsilon=0.01) 
epsilon_greedy_solver.run(150000) 
print('epsilon-贪婪算法的累积懊悔为：', epsilon_greedy_solver.regret) 
plot_results([epsilon_greedy_solver], ["EpsilonGreedy"])

# 不同e值的影响
# np.random.seed(0) 
# epsilons = [1e-4, 0.01, 0.1, 0.25, 0.5] 
# epsilon_greedy_solver_list = [EpsilonGreedy(bandit_10_arm, epsilon=e) for e in epsilons] 
# epsilon_greedy_solver_names = ["epsilon={}".format(e) for e in epsilons]
# for solver in epsilon_greedy_solver_list: 
#     solver.run(5000) 
# plot_results(epsilon_greedy_solver_list, epsilon_greedy_solver_names)

#试e值随时间衰减的e-贪婪算法
class DecayingEpsilonGreedy(Solver): 
    """ epsilon 贪婪算法,继承 Solver 类 """ 
    def __init__(self, bandit,init_prob=1.0):
        super(DecayingEpsilonGreedy, self).__init__(bandit) 
        #初始化拉动所有拉杆的期望奖励估值
        self.estimates = np.array([init_prob] * self.bandit.k) 
        self.total_count =0
    def run_one_step(self): 
        self.total_count += 1
        if np.random.random() < 1/self.total_count: 
            k = np.random.randint(0, self.bandit.k) # 随机选择一根拉杆
        else: 
            k = np.argmax(self.estimates) # 选择期望奖励估值最大的拉杆
        r = self.bandit.step(k) # 得到本次动作的奖励
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k]) 
        return k


np.random.seed(1) 
decaying_epsilon_greedy_solver = DecayingEpsilonGreedy(bandit_10_arm)
decaying_epsilon_greedy_solver.run(150000) 
print('epsilon 值衰减的贪婪算法的累积懊悔为：', decaying_epsilon_greedy_solver.regret) 
plot_results([decaying_epsilon_greedy_solver], ["DecayingEpsilonGreedy"])

class UCB(Solver): 

    def __init__(self, bandit, coef, init_prob=1.0): 
        super(UCB, self).__init__(bandit) 
        self.total_count = 0 
        self.estimates = np.array([init_prob] * self.bandit.k) 
        self.coef = coef #control ua
    def run_one_step(self): 
        self.total_count += 1 
        ucb = self.estimates + self.coef * np.sqrt(np.log(self.total_count) / (2 * (self.counts + 1))) # 计算上置信界
        k = np.argmax(ucb) # 选出上置信界最大的拉杆
        r = self.bandit.step(k) 
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k]) 
        return k 
np.random.seed(1) 
coef = 1 # 控制不确定性比重的系数
UCB_solver = UCB(bandit_10_arm, coef) 
UCB_solver.run(150000) 
print('上置信界算法的累积懊悔为：', UCB_solver.regret) 
plot_results([UCB_solver], ["UCB"])

class ThompsonSampling(Solver): 

    def __init__(self, bandit): 
        super(ThompsonSampling, self).__init__(bandit) 
        self._a = np.ones(self.bandit.k) # 列表,表示每根拉杆奖励为 1 的次数
        self._b = np.ones(self.bandit.k) # 列表,表示每根拉杆奖励为 0 的次数
    def run_one_step(self): 
        samples = np.random.beta(self._a, self._b) # 按照 Beta 分布采样一组奖励样本
        k = np.argmax(samples) # 选出采样奖励最大的拉杆
        r = self.bandit.step(k) 
        self._a[k] += r # 更新 Beta 分布的第一个参数
        self._b[k] += (1 - r) # 更新 Beta 分布的第二个参数
        return k 
np.random.seed(1) 
thompson_sampling_solver = ThompsonSampling(bandit_10_arm) 
thompson_sampling_solver.run(150000) 
print('汤普森采样算法的累积懊悔为：', thompson_sampling_solver.regret) 
plot_results([thompson_sampling_solver], ["ThompsonSampling"])