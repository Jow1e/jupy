import numpy as np
import matplotlib.pyplot as plt
import jupy as jp


x1 = np.random.normal(0, 2, (60, 2))
x1[np.abs(x1) < 2] *= 4

y1 = np.zeros(60)

x2 = np.random.normal(0, 1, (10, 2))
y2 = np.ones(10)

x1 = jp.Tensor(x1, dtype=float)
y1 = jp.Tensor(y1, dtype=int)

x2 = jp.Tensor(x2, dtype=float)
y2 = jp.Tensor(y2, dtype=int)

model = jp.Sequential(
	jp.Linear(2, 8),
	jp.PReLU(8),
	jp.Linear(8, 8),
	jp.PReLU(8),
	jp.Linear(8, 8)
)

optimizer = jp.Adam(model.parameters(), lr=0.05, weight_decay=1e-3)

for i in range(1000):
	p1 = model(x1)
	p2 = model(x2)
	
	loss1 = jp.cross_entropy(p1, y1)
	loss2 = jp.cross_entropy(p2, y2)
	
	loss = loss1 + loss2
	
	loss.backward()
	
	optimizer.step()
	optimizer.reset_grad()
	
	print(f"Epoch {i + 1}")
	print(f"Loss = {loss}")
	print()

test = jp.Tensor(np.random.uniform(-10, 10, (1000, 2)))
output = model(test).data.argmax(axis=1)

test0 = test.data[output == 0]
test1 = test.data[output == 1]

plt.scatter(*test0.T, alpha=0.2, c='green')
plt.scatter(*test1.T, alpha=0.2, c='blue')

plt.scatter(*x1.data.T, c='green')
plt.scatter(*x2.data.T, c='blue')

plt.show()
