# Завдання 1

import pulp

model = pulp.LpProblem("Maximaze_Drink", pulp.LpMaximize)

limonade = pulp.LpVariable('limonade', lowBound=0, cat='Continuous')
fruit_juice = pulp.LpVariable('fruit_juice', lowBound=0, cat='Continuous')

model += limonade + fruit_juice, "total_drink_count"  #максимізувати x + y

model += 2 * limonade + 1 * fruit_juice <= 100, "Water_Limit"
model += 1 * limonade <= 50, "Sugar_Limit"
model += 1 * limonade <= 30, "Lemon_juice_Limit"
model += 2 * fruit_juice <= 40, "Fruit_puree_Limit"

model.solve()

print("status:",pulp.LpStatus[model.status])

# Вивід результатів
print("Виробили Лимонад:", limonade.varValue)
print("Вироблили Фруктовий сік:", fruit_juice.varValue)

# Завдання 2

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as spi

a, b = 0, 2
num_samples = 100000

# Визначення функції та межі інтегрування
def f(x):
    return x ** 2

def visualize_results(x_random, y_random, a, b, f):
    # Створення діапазону значень для x
    x = np.linspace(-0.5, 2.5, 400)
    y = f(x)

    _, ax = plt.subplots()

    # Малювання функції
    ax.plot(x, y, 'r', linewidth=2)
    ax.scatter(x_random, y_random, color="red", s=1, alpha=0.1)

    # Заповнення області під кривою
    ix = np.linspace(a, b)
    iy = f(ix)
    ax.fill_between(ix, iy, color='gray', alpha=0.3)

    # Налаштування графіка
    ax.set_xlim([x[0], x[-1]])
    ax.set_ylim([0, max(y) + 0.1])
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')

    # Додавання меж інтегрування та назви графіка
    ax.axvline(x=a, color='gray', linestyle='--')
    ax.axvline(x=b, color='gray', linestyle='--')
    ax.set_title('Графік інтегрування f(x) = x^2 від ' + str(a) + ' до ' + str(b))

    plt.grid()
    plt.show()

def monte_carlo(a, b, num_samples):
    x_random = np.random.uniform(a, b, num_samples)
    y_random = np.random.uniform(0, f(b), num_samples)
    
    # Знаходження кількості точок під кривою
    under_curve = y_random < f(x_random)
    area_under_curve = np.sum(under_curve) / num_samples * (b - a) * f(b)
    
    # Обчислення інтеграла за допомогою SciPy
    result, error = spi.quad(f, a, b)
    
    print('Площа, обчислена методом Монте-Карло:', area_under_curve)
    print('Площа, обчислена з використанням scipy.integrate.quad:', result)
    
    visualize_results(x_random, y_random, a, b, f)

# Виклик функції для обчислення площі
monte_carlo(a, b, num_samples)


