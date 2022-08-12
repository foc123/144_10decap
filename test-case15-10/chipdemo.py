import os,math
import random
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as  nn
import torch.nn.functional as F

node_sum = 15
node_cs = 10
lr = 0.001
gamma = 0.9
epsilon = 0.95
loc_num = [416, 1250, 2083, 2916, 3750, 4583, 5416, 6250, 7083, 7916, 8750, 9583]
loc_node = []


#train
memorysize = 3000 
batch_size = 32
itertimes = 20

# debug
#memorysize = 10
#batch_size = 4
#itertimes = 3



def jc(n):
    res = 1
    for i in range(1,n+1):
        res *= i
    return res

for i in loc_num:
    for j in loc_num:
        loc_node.append([j,i])

#loc_node = np.reshape(loc_node,[12,12])

#print(loc_node)
#print(loc_node[12][1],loc_node[12][0])

# def adddecap():
#     with open('vdd_decap.1','w') as f1:
#         f1.write('%s',str1)
#     f1.close()


def readresult(filename):
    a1 = np.genfromtxt(filename)
    return a1

def reset_dc():
    with open('vdd_decap.1','w') as f:
        f.write('')
    f.close()

def readvdi(file):  # 读取csv中的vdi的数据，得到的是array数据，这里作为input
    zvdi = readresult(file)
    z = zvdi[:,2]
    return z


def location(file):         #节点位置
    a = readresult(file)
    x = a[:,0]
    y = a[:,1]
    loc = []
    for r in range(len(a)):
        loc.append([x[r],y[r]])
    return loc


#z = readvdi('chiplet1_vdd_1_vdi.csv')   # 读取电路vdi的所有节点数据，为np.array

# 获取VDI最大的十二个点的坐标

#print('idx:',idx)

#运行程序获得VDI分布图,即得到csv数据

def run_os():
    os.system('ngspice -b interposer1_tr.sp -r interposer1_tr.raw')
    os.system('bin/inttrvmap int1.conf interposer1_tr.raw 1.0 0.05')


def target_vdi():
    os.system('bin/diedcapgen 10 1e-9 chiplet1_vdd.decap vdd_decap.1')
    run_os()
    a = readvdi('chiplet1_vdd_1_vdi.csv')
    sum_a = np.sum(a)
    return sum_a 

reset_dc()
run_os()
z = readvdi('chiplet1_vdd_1_vdi.csv')
z1 = np.sort(z)[-15:]   # 
# print(z1)
idx = []
for i in range(144):
    if z[i] in z1:
        idx.append(i)



loss_val = []
nstate = len(loc_node)
naction = int(math.factorial(node_sum)/(math.factorial(node_cs) * math.factorial(node_sum - node_cs)))
set0 = []
for i in range(node_sum):
    for j in range(i+1,node_sum):
         for k in range(j+1,node_sum):
              for m in range(k+1,node_sum):
                   for n in range(m+1,node_sum):
                        set0.append([i,j,k,m,n])
# set0是指12cap中选择两个cap为0，即在12个位置中选择十个位置放置decap


class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(nstate,500)
        self.fc1.weight.data.normal_(0,0.1)
        self.fc2 = nn.Linear(500,5000)
        self.fc2.weight.data.normal_(0,0.1)
        self.out = nn.Linear(5000,naction)
        self.out.weight.data.normal_(0,0.1)

    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        action_val = self.out(x)
        return action_val


class ChipPdn:

    def __init__(self):
        self.eval_net,self.target_net = Net(),Net()
        self.memorysize = memorysize
        self.memory = np.zeros([memorysize, nstate * 2 + 2])
        self.mem_cnt = 0
        self.loss_func = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(),lr=lr)
        self.learn_step_counter = 0

    def step(self,action):      # 选择动作之后，进行数据的读取，输出变化之后的vdi分布
        s0 = set0[action]
        idx1 = idx
        val = [1] * node_sum
        for x in s0:
            val[x] = 0
        for t in range(len(val)):
            val[t] *= 1e-9
       # del idx1[s0[1]]
       # del idx1[s0[0]]
        str1 = ''
        for i,ee  in enumerate(idx1):
       # for ee in idx1:
            if val[i] > 0 :
                str1 = str1 + 'c_nd_1_0_%d_%d nd_1_0_%d_%d 0 %e\n' %(
                        loc_node[ee][0], loc_node[ee][1], loc_node[ee][0], loc_node[ee][1],val[i])
        f = open('vdd_decap.1', 'w')
        f.write(str1)
        f.close()
        run_os()
        state_ = readvdi('chiplet1_vdd_1_vdi.csv')
        return state_

    def csact(self,x):          # 选择动作函数
        x = torch.unsqueeze(torch.FloatTensor(x),0)
        if random.random() < epsilon:
            action = random.randint(0,naction - 1)  # 80% exploration is <
        else:
            action_val = self.eval_net.forward(x)   # 20% explotition
            action = torch.max(action_val,1)[1].data.numpy()
            action = action[0]
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.mem_cnt % memorysize
        self.memory[index, :] = transition
        self.mem_cnt += 1

   # def reset(self,cap,val):        # cap是随机增加的cap数目，val是每个cap的容值
        #cap_val = []
    def reset(self):
        st = ''
        with open('vdd_decap.1','w') as f1 :
            f1.write('')
        f1.close()
       # with open('random_cap','w') as f2:
         #   ix = random.sample(range(144),cap)
         #   for ai in ix:
          #      st = st + 'c_val_%d_%d nd_1_0_%d_%d 0 %e\n' % (
        #            loc_node[ai][0], loc_node[ai][1], loc_node[ai][0], loc_node[ai][1],val)
         #   f2.write(st)
       # f2.close()
        run_os()
        vdi = readvdi('chiplet1_vdd_1_vdi.csv')
        return vdi

    def learn(self):
        # target parameter update
        if self.learn_step_counter % itertimes == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(memorysize, batch_size)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :nstate])
        b_a = torch.LongTensor(b_memory[:, nstate:nstate+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, nstate+1:nstate+2])
        b_s_ = torch.FloatTensor(b_memory[:, -nstate:])

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + gamma * q_next.max(1)[0].view(batch_size, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)
        loss_val.append(loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# debug 
#episodes = 4
#looptimes = 5

decay = 0.04
# train
episodes = 100
looptimes = 50


cp = ChipPdn()
rewardlist = []
ze = []
maxlist = []
minlist = []
min_r = []
max_r = []
target_v = target_vdi()
#os.system('python3 utils/pltdroopint.py chiplet1_vdd_1_vdi.csv')
for eps in range(episodes):
    t = 0
    max_s = 0
    ep_r = 0
    r_min = 1
    r_max = -1
    print('-' * 30)
    print('eps:',eps)
    s = cp.reset()
    init_vdi = np.sum(s)
    min_s = init_vdi
    while t < looptimes:
        zero = 0
        t += 1
        a = cp.csact(s)
        s_ = cp.step(a)
       # r = 1 - np.sum(s_) / init_vdi if np.sum(s_) < 0.3*init_vdi else -np.sum(s_) / init_vdi
        r = (target_v - np.sum(s_)) / (target_v +np.sum(s_))
        cp.store_transition(s,a,r,s_)
        ep_r += r
        print(np.sum(s),a,r,np.sum(s_))
        if cp.mem_cnt > memorysize:
            cp.learn()
        per_ = readvdi('chiplet1_vdd_1_vdi.csv')
        for l in per_:
            if l == 0 :
                zero += 1
        ze.append(zero)
        if np.sum(s_) < min_s:
            min_s = np.sum(s_)
        if np.sum(s_) > max_s:
            max_s = np.sum(s_)
        s = s_
        if r > r_max:
            r_max = r
        if r < r_min :
            r_min = r


    epsilon = 0.95 * np.exp(decay * eps)
    rewardlist.append(ep_r)
    minlist.append(min_s)
    maxlist.append(max_s)
    min_r.append(r_min)
    max_r.append(r_max)




loss_val = torch.tensor(loss_val)
plt.plot([a for a in range(len(loss_val))],loss_val)
plt.title('loss function')
plt.xlabel('episodes')
plt.ylabel('value')
plt.figure()
plt.plot([b for b in range(len(rewardlist))],rewardlist)
plt.title('reward')
plt.figure()
plt.plot([c for c in range(len(ze))],ze)
plt.title('No. of no-violated')
plt.xlabel('step')
plt.figure()
plt.plot([d for d in range(len(maxlist))],maxlist,label='max VDI')
plt.plot([e for e in range(len(minlist))],minlist,label='min VDI')
plt.plot([0,episodes - 1],[target_v,target_v],label='target VDI',linestyle='--')
plt.legend()
plt.figure()
plt.plot([d for d in range(len(max_r))],max_r,label='max reward ')
plt.plot([e for e in range(len(min_r))],min_r,label='min reward')
plt.title('max/min reward of per episodes')
plt.xlabel('episodes')
plt.ylabel('reward value')
plt.legend()
plt.show()

# test example
tes = cp.reset()
test1 = readvdi('chiplet1_vdd_1_vdi.csv')
tes = torch.unsqueeze(torch.FloatTensor(tes),0)
action_val = cp.eval_net.forward(tes)   
action = torch.max(action_val,1)[1].data.numpy()
action = action[0]
tes_ = cp.step(action)
print('test init:%e,action:%d,test end:%e' % (np.sum(test1),action,np.sum(tes_)) )

os.system('python3 utils/pltvdroopint.py chiplet1_vdd_1_vdi.csv')

for m in idx:
    print('decap location:',loc_node[m])
print('z1:',z1)




