import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq

#Least squares method with scipy.optimize
def fitLinear(p, x, y):
    return p[0]*x + p[1] - y

def gen_rand(low, up, size):
	noZeroList = []
	while len(noZeroList) < size:
		tmpFloat = random.uniform(low, up)
		if tmpFloat != 0:
			noZeroList.append(tmpFloat)
	return noZeroList

def gen_gaussRand(mu, sigma, size):
	noZeroList = []
	while len(noZeroList) < size:
		tmpFloat = random.gauss(mu, sigma)
		if tmpFloat != 0:
			noZeroList.append(tmpFloat)
	return noZeroList

def arithMean(list):
    avg = sum(list) / len(list)
    return avg

sell = 90 # v
life = 180 # S
inventory = sell*life/3 # V
rules = list(range(2, 4)) # [n_1, ..., n_-1]
deadlines = [int(life/rules[i]) for i in range(len(rules))] # [S/n_1, ..., S/n_-1]
#print(deadlines)

#random.seed(29)
sellRatios = [gen_gaussRand(1, 0.3, deadlines[i]) for i in range(len(rules))]
#print(sellRatios)

sizeOfTest = 1000
predSellRatiosList = [[gen_rand(0.2, 1.8, sizeOfTest) for i in range(deadlines[j])] for j in range(len(rules))]

predErrorList = []
for j in range(len(rules)): # rules[j] corresponds to '1/n rule'
	predError = []
	for i in range(sizeOfTest):
		tmpList = []
		for k in range(deadlines[j]): # predError avg. over deadlines days
			tmpList.append(predSellRatiosList[j][k][i] / sellRatios[j][k])
		predError.append(arithMean(tmpList))
	predErrorList.append(predError)
#print(predErrorList)

lossList = [[inventory - sell*sum([predSellRatiosList[j][k][i] for k in range(deadlines[j])]) for i in range(sizeOfTest)] for j in range(len(rules))]
#print(lossList)

colorStr = ['red', 'green']
for i in range(len(rules)):
	plt.scatter(predErrorList[i], lossList[i], alpha=0.5)
	# linear regression for each rule ->
	p0 = [0, 0] # initial (k, b)
	ret = leastsq(fitLinear, p0, args=(np.array(predErrorList[i]), np.array(lossList[i])))
	k, b = ret[0]
	x = np.linspace(min(predErrorList[i]), max(predErrorList[i]), 1000)
	y = k * x + b
	plt.plot(x, y, color=colorStr[i], alpha=0.8, label='y='+str(round(k,2))+'*x+'+str(round(b,2)))
	# <- linear regression end
plt.xlabel("Prediction error")
plt.xlim([0.7, 1.4])
plt.ylabel("Loss")
plt.ylim([-14500, 2500])
plt.legend(loc='upper right')
plt.scatter([1], [0], marker='*', color='black')
plt.grid()
plt.show()














