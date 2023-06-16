# importing Python library
import numpy as np
from matplotlib import pyplot as plt

def StepFunc(v):
	if v >= 0:
		return 1
	else:
		return 0

def PModel(x, w, b):
	v = np.dot(w, x) + b
	y = StepFunc(v)
	return y

def NotFunc(x):
	wNOT = -1
	bNOT = 0.5
	return PModel(x, wNOT, bNOT)

def AndFunc(x):
	w = np.array([1, 1])
	bAND = -1.5
	return PModel(x, w, bAND)

def OrFunc(x):
	w = np.array([1, 1])
	bOR = -0.5
	return PModel(x, w, bOR)

def XorFunc(x):
	y1 = AndFunc(x)
	y2 = OrFunc(x)
	y3 = NotFunc(y1)
	x1 = np.array([y2, y3])
	y1 = AndFunc(x1)
	return y1

test1 = np.array([0, 0])
test2 = np.array([0, 1])
test3 = np.array([1, 0])
test4 = np.array([1, 1])

print("XOR({}, {}) = {}".format(0, 0, XorFunc(test1)))
print("XOR({}, {}) = {}".format(0, 1, XorFunc(test2)))
print("XOR({}, {}) = {}".format(1, 0, XorFunc(test3)))
print("XOR({}, {}) = {}".format(1, 1, XorFunc(test4)))

fig = plt.xkcd()

plt.scatter(0, 0, s=50, color="red", zorder=3)
plt.scatter(0, 1, s=50, color="green", zorder=3)
plt.scatter(1, 0, s=50, color="green", zorder=3)
plt.scatter(1, 1, s=50, color="red", zorder=3)

plt.xlim(-1, 2)
plt.ylim(-1, 2)

x = np.linspace(-2, 2, 100)
y = -x + 0.5

xx = np.linspace(-2, 2, 100)
yy = -x + 1.5

plt.xlabel("Input 1")
plt.ylabel("Input 2")
plt.title("State Space of Input Vector")

plt.grid(True, linewidth=1, linestyle=':')
plt.tight_layout()

plt.plot(x,y, label = "y = -x + 0.5")

plt.plot(xx,yy, label = "y = -x + 1.5")

plt.legend()
plt.show()
