
# coding: utf-8

# In[1]:

import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np


# In[2]:

data = pd.read_csv('dataset/family_data.csv')


# In[3]:

sub = pd.read_csv('submit2_4.csv')


# In[4]:

# Create a new model
m = gp.Model("santa")


# In[5]:

m.setParam('MIPFocus',2)
m.setParam('MIPGap',0)
m.setParam('MIPGapAbs',0)
# m.setParam('Cutoff',69762.0)
# m.setParam('Threads',5)


# In[6]:

MAX_OCCUPANCY = 300
MIN_OCCUPANCY = 125
days = range(1,101)
count = range(5000)


# In[7]:

C = np.zeros((5000,101))
choices = data[['choice_'+str(i) for i in range(10)]].values
for f in range(5000):
    for d in range(1,101):
        l = list(choices[f])
        if d in l:
            if l.index(d) == 0:
                C[f,d] = 0
            elif l.index(d) == 1:
                C[f,d] = 50
            elif l.index(d) == 2:
                C[f,d] = 50 + 9 * data.n_people[f]
            elif l.index(d) == 3:
                C[f,d] = 100 + 9 * data.n_people[f]
            elif l.index(d) == 4:
                C[f,d] = 200 + 9 * data.n_people[f]
            elif l.index(d) == 5:
                C[f,d] = 200 + 18 * data.n_people[f]
            elif l.index(d) == 6:
                C[f,d] = 300 + 18 * data.n_people[f]
            elif l.index(d) == 7:
                C[f,d] = 300 + 36 * data.n_people[f]
            elif l.index(d) == 8:
                C[f,d] = 400 + 36 * data.n_people[f]
            elif l.index(d) == 9:
                C[f,d] = 500 + 235 * data.n_people[f]
        else:
            C[f,d] = 500 + 434 * data.n_people[f]
            
A = np.zeros((176,176))
A100 = np.zeros((176,))
for i in range(125,301):
    for j in range(125,301):
        A[i-125][j-125] = ((i-125)*(i**(0.5+abs(i-j)/50)))/400

for i in range(125,301):
    A100[i-125] = ((i-125) * (i ** 0.5))/400


# In[8]:

# Create variables
F = [[m.addVar(vtype=GRB.BINARY) for i in range(101)] for j in range(5000)]


# In[9]:

m.update()


# In[10]:

Y = [[[m.addVar(vtype=GRB.BINARY) for i in range(101)] for j in range(176)] for i in range(176)]


# In[11]:

L = np.zeros((5000,101))
L2 = np.zeros((176,176,101))


# In[12]:

occ = np.zeros(101)
for i,val in enumerate(sub.assigned_day.values):
    occ[val] += data.at[i,'n_people']


# In[13]:

for i in range(5000):
    for j in range(1,101):
        if j == sub.at[i,'assigned_day']:
            F[i][j].start=1
            L[i][j]=1
        else:
            F[i][j].start=0
            L[i][j]=0


# In[14]:

for i in range(176):
    for k in range(176):
        for j in range(101):
            Y[i][k][j].start=0
            L2[i][k][j]=0
for i in days:
    if i != 100:
        Y[int(occ[i]-125)][int(occ[i+1]-125)][i].start=1
        L2[int(occ[i]-125)][int(occ[i+1]-125)][i]=1
    else:
        Y[int(occ[i]-125)][int(occ[i]-125)][i].start=1
        L2[int(occ[i]-125)][int(occ[i]-125)][i]=1


# In[15]:

print(sum(L[f][d]*C[f,d] for f in count for d in days))


# In[16]:

print(sum(L2[i][j][d] * A[i,j] for i in range(176) for j in range(176) for d in range(1,100)))


# In[17]:

print(sum(L2[i][j][100]*A100[i] for i in range(176) for j in range(176)))


# In[18]:

N = []
P = []
Q = []
for d in days:
    N.append(sum(F[f][d]*data.n_people[f] for f in count))
    P.append(sum(Y[i][j][d]*(i+125) for i in range(176) for j in range(176)))
    Q.append(sum(Y[i][j][d]*(j+125) for i in range(176) for j in range(176)))


# In[19]:

m.addConstrs(N[d-1] <= MAX_OCCUPANCY for d in days)
m.addConstrs(N[d-1] >= MIN_OCCUPANCY for d in days)
m.addConstrs(sum(F[f][d] for d in days) == 1 for f in count)
m.addConstrs(P[d-1]==N[d-1] for d in days)
m.addConstrs(Q[d]==N[d] for d in range(99,100))
m.addConstrs(Q[d-1]==N[d] for d in range(1,100))
m.addConstrs(sum(Y[i][j][d] for i in range(176) for j in range(176))==1 for d in days)


# In[20]:

pcost = sum(F[f][d]*C[f,d] for f in count for d in days)


# In[21]:

acost1 = sum(Y[i][j][d] * A[i,j] for i in range(176) for j in range(176) for d in range(1,100))


# In[22]:

acost2 = sum(Y[i][j][100]*A100[i] for i in range(176) for j in range(176))


# In[23]:

m.addConstr(pcost < 63100.0)
# m.addConstr(acost1+acost2 >= 6020.043431)


# In[24]:

m.setObjective(pcost+acost1+acost2,GRB.MINIMIZE)


# In[25]:

logfile = open('cb.log', 'w')


# In[26]:

m._logfile = logfile
m._vars = m.getVars()
m._sub = sub
m._sub_id = 0


# In[27]:

def mycallback(model, where):
    if where == GRB.Callback.MIPSOL:
        # MIP solution callback
        nodecnt = model.cbGet(GRB.Callback.MIPSOL_NODCNT)
        obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
        solcnt = model.cbGet(GRB.Callback.MIPSOL_SOLCNT)
        x = model.cbGetSolution(model._vars)
        tp = np.zeros((5000,101))
        i=0
        for v in x:
            tp[i//101][i%101]=v
            i = i+1
        model._sub.assigned_day = np.argmax(tp,1)
        model._sub.to_csv(f'submit{m._sub_id}.csv',index=False)
        m._sub_id = m._sub_id + 1
#         print('**** New solution at node %d, obj %g, sol %d, '
#               'x[0] = %g ****' % (nodecnt, obj, solcnt, x[0]))
    elif where == GRB.Callback.MESSAGE:
        # Message callback
        msg = model.cbGet(GRB.Callback.MSG_STRING)
        model._logfile.write(msg)


# In[28]:

m.optimize(mycallback)

print('')
print('Optimization complete')
if m.SolCount == 0:
    print('No solution found, optimization status = %d' % m.Status)
else:
    print('Solution found, objective = %g' % m.ObjVal)

logfile.close()


# In[29]:

tp = np.zeros((5000,101))
i=0
for v in m.getVars():
    if v.X != 0.0:
        tp[i//101][i%101]=1
    i = i+1

sub.assigned_day = np.argmax(tp,1)


# In[30]:

sub.to_csv('submission.csv',index=False)


# In[31]:

# for v in m.getVars():
#     print('%s %g' % (v.varName, v.x))

# print('Obj: %g' % acost1.getValue())
# print('Obj: %g' % acost2.getValue())
# print('Obj: %g' % pcost.getValue())


# In[ ]:




# In[ ]:



