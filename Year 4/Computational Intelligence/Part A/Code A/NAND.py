import numpy
from matplotlib import pyplot as plt

def StepFunc(v):
	if v >= 0:
		return 1
	else:
		return 0

def PModel(x, w, b):
	v = numpy.dot(w, x) + b
	y = StepFunc(v)
	return y

def NotFunc(x):
	wNOT = -1
	bNOT = 0.5
	return PModel(x, wNOT, bNOT)

def AndFunc(x):
	weights = numpy.array([1, 1])
	bAND = -1.5
	return PModel(x, weights, bAND)

def NandFunc(x):
	OutAnd = AndFunc(x)
	OutNot = NotFunc(OutAnd)
	return OutNot

X1 = numpy.array([0, 0])
X2 = numpy.array([0, 1])
X3 = numpy.array([1, 0])
X4 = numpy.array([1, 1])

print("NAND({}, {}) = {}".format(0, 0, NandFunc(X1)))
print("NAND({}, {}) = {}".format(0, 1, NandFunc(X2)))
print("NAND({}, {}) = {}".format(1, 0, NandFunc(X3)))
print("NAND({}, {}) = {}".format(1, 1, NandFunc(X4)))

fig = plt.xkcd()

plt.scatter(0, 0, s=50, color="green", zorder=3)
plt.scatter(0, 1, s=50, color="green", zorder=3)
plt.scatter(1, 0, s=50, color="green", zorder=3)
plt.scatter(1, 1, s=50, color="red", zorder=3)

plt.xlim(-1, 2)
plt.ylim(-1, 2)

x = numpy.linspace(-2, 2, 100)
y = -x + 1.5

plt.xlabel("Input 1")
plt.ylabel("Input 2")
plt.title("State Space of Input Vector")

plt.grid(True, linewidth=1, linestyle=':')
plt.tight_layout()

plt.plot(x,y, label = "y = -x + 1.5")
plt.legend()

plt.show()