import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


x1 = np.array([0, 0, 0, 1, 1, 2, 2, 2])
x2 = np.array([1.5, 2.5, 3.5, 1.5, 3.5, 1.5, 2.5, 2.5])
y = np.array([2.3, 7.3, 0.9, 2.8, 1.8, 8.3, 5.4, 7.2])

n = len(y)  # 8


# 1. Обчислення сум для системи нормальних рівнянь МНК
sum_x1    = np.sum(x1)
sum_x2    = np.sum(x2)
sum_y     = np.sum(y)
sum_x1_sq = np.sum(x1**2)
sum_x2_sq = np.sum(x2**2)
sum_x1x2  = np.sum(x1 * x2)
sum_x1y   = np.sum(x1 * y)
sum_x2y   = np.sum(x2 * y)

print("Система нормальних рівнянь:")
print(f"{n} a0 + {sum_x1} a1 + {sum_x2} a2 = {sum_y}")
print(f"{sum_x1} a0 + {sum_x1_sq} a1 + {sum_x1x2} a2 = {sum_x1y}")
print(f"{sum_x2} a0 + {sum_x1x2} a1 + {sum_x2_sq} a2 = {sum_x2y}")

# 2. Розв’язання системи методом Гаусса 

def gauss_elimination(A, B):
    n = len(B)
    AB = np.hstack([A, B.reshape(-1, 1)])  # розширена матриця

    
    for i in range(n):
        #  вибір головного елемента
        max_row = i + np.argmax(np.abs(AB[i:, i]))
        AB[[i, max_row]] = AB[[max_row, i]]

        # Нормалізація рядка
        if abs(AB[i, i]) < 1e-10:
            raise ValueError("Матриця вироджена або погано обумовлена")
        AB[i] = AB[i] / AB[i, i]

        # Обнулення нижче діагоналі
        for j in range(i + 1, n):
            AB[j] -= AB[i] * AB[j, i]

    # Зворотний хід
    X = np.zeros(n)
    for i in range(n - 1, -1, -1):
        X[i] = AB[i, -1] - np.dot(AB[i, i + 1:n], X[i + 1:n])

    return X

# Матриця коефіцієнтів і вільні члени
A = np.array([
    [n, sum_x1, sum_x2],
    [sum_x1, sum_x1_sq, sum_x1x2],
    [sum_x2, sum_x1x2, sum_x2_sq]
], dtype=float)

B = np.array([sum_y, sum_x1y, sum_x2y], dtype=float)

# Розв’язок
coeff = gauss_elimination(A, B)
a0, a1, a2 = coeff

print("\nКоефіцієнти моделі:")
print(f"a0 = {a0:.4f}")
print(f"a1 = {a1:.4f}")
print(f"a2 = {a2:.4f}")
print(f"Залежність: y ≈ {a0:.4f} + {a1:.4f} x1 + {a2:.4f} x2")


# 3. Значення в точці (1.5, 3)

x1_point = 1.5
x2_point = 3.0
y_point = a0 + a1 * x1_point + a2 * x2_point
print(f"\nЗначення функції в точці (x1={x1_point}, x2={x2_point}): y = {y_point:.4f}")


# 4. Обчислення R²

y_pred = a0 + a1 * x1 + a2 * x2          # прогнозовані значення
y_mean = np.mean(y)                      
ss_res = np.sum((y - y_pred)**2)         # сума квадратів залишків
ss_tot = np.sum((y - y_mean)**2)         # загальна сума квадратів

r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
print(f"\nКоефіцієнт детермінації R² = {r2:.4f} ({r2*100:.2f} %)")
print(f"Модель пояснює {r2*100:.1f} % варіації даних y")


# 5. Побудова 3D-графіка

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Сітка для площини 
x1_grid = np.linspace(0, 2, 20)
x2_grid = np.linspace(1.5, 3.5, 20)
X1, X2 = np.meshgrid(x1_grid, x2_grid)
Y = a0 + a1 * X1 + a2 * X2

# Площина
ax.plot_surface(X1, X2, Y, alpha=0.5, cmap='coolwarm', label='Площина МНК')

# Точки даних (червоні)
ax.scatter(x1, x2, y, color='red', s=60, label='Дані з таблиці')

# Шукана точка (зелена)
ax.scatter(x1_point, x2_point, y_point, color='green', s=150, label='Точка (1.5, 3)')

ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
ax.set_title('Лінійна модель МНК: y ≈ {:.3f} + {:.3f} x1 - {:.3f} x2\nR² = {:.3f}'.format(a0, a1, -a2, r2))
ax.legend()

plt.show()