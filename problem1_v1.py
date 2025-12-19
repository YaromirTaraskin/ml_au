import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print('''

######################################
#  ИНФОРМАЦИЯ ПО ЛИНЕЙНОЙ РЕГРЕССИИ  #
######################################
''')
print()
print("――――――――――――――――")
print("1. РАЗМЕР БАТЧА:")
print("――――――――――――――――")
print("Batch size 1 (стохастический): Быстрая сходимость в начале, но сильные колебания")
print("Batch size 8-32 (мини-батч): Хороший баланс между скоростью и стабильностью")
print("Batch size N (полный GD): Плавная сходимость, но медленнее и требовательнее к памяти")
print()
print("―――――――――――――――――――――――――――")
print("2. LEARNING RATE SCHEDULES:")
print("―――――――――――――――――――――――――――")
print("Постоянный LR: Простота, но может не сходиться или сходиться медленно")
print("Экспоненциальный: Плавное уменьшение, хорош для тонкой настройки")
print("Ступенчатый: Резкие изменения, может ускорить сходимость")
print()
print("――――――――――――――――")
print("3. ОПТИМИЗАТОРЫ:")
print("――――――――――――――――")
print("SGD: Базовый вариант, требует тщательного подбора LR")
print("SGD + Momentum: Лучше проходит через локальные минимумы")
print("SGD + Nesterov: Более точное обновление с учетом будущего градиента")
print("Adam: Адаптивный LR, быстрая сходимость, хорош по умолчанию")
# noinspection SpellCheckingInspection
print("RMSprop: Хорош для задач с разреженными градиентами")
print()
print("――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――")
print("4. ДЛЯ ЛИНЕЙНОЙ РЕГРЕССИИ СУЩЕСТВУЮТ СЛЕДУЮЩИЕ РЕКОМЕНДАЦИИ:")
print("――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――")
print("1. Начинать с батча размером 16-32")
print("2. Использовать Adam как оптимизатор по умолчанию")
print("3. Для больших данных использовать увеличенный размер батча")
print("4. Экспоненциальное затухание LR часто работает лучше постоянного")
print("5. Всегда нормализовывать данные перед обучением")

print('''

###############
#  ОБРАБОТКА  #
###############
''')

print("Загрузка и подготовка данных Wine Quality...")
data_csv_url = \
    "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
datFra = pd.read_csv(data_csv_url, sep=';')

# Разделение на признаки и целевую переменное
X = datFra.drop(labels='quality', axis=1).values
y = datFra['quality'].values.reshape(-1, 1)

# Нормализация данных
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y)

# Добавление свободного члена (intercept)
X = np.hstack([np.ones((X.shape[0], 1)), X])

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Размер обучающей выборки: {X_train.shape[0]} образцов")
print(f"Размер тестовой выборки: {X_test.shape[0]} образцов")
print(f"Количество признаков: {X_train.shape[1] - 1} (плюс свободный член)\n")


class LinearRegressionSGD:
    """Реализация SGD с различными размерами батча"""

    def __init__(self, n_features, learning_rate=0.01, batch_size_=32):
        self.theta = np.random.randn(n_features, 1) * 0.01
        self.learning_rate = learning_rate
        self.batch_size = batch_size_
        self.loss_history = []

    def compute_loss(self, X_, y_):
        m = len(y_)
        predictions = X_.dot(self.theta)
        loss_ = (1 / (2 * m)) * np.sum((predictions - y_) ** 2)
        return loss_

    def fit(self, X_, y_, n_epochs=100, learning_rate_schedule=None, verbose=False):
        m = len(y_)

        for epoch_ in range(n_epochs):
            if learning_rate_schedule:
                lr = learning_rate_schedule(epoch_, self.learning_rate)
            else:
                lr = self.learning_rate

            indices = np.random.permutation(m)
            # noinspection PyPep8Naming
            X_shuffled = X_[indices]
            y_shuffled = y_[indices]

            for i in range(0, m, self.batch_size):
                # noinspection PyPep8Naming
                X_batch = X_shuffled[i:i + self.batch_size]
                y_batch = y_shuffled[i:i + self.batch_size]

                predictions = X_batch.dot(self.theta)
                gradient_mse = X_batch.T.dot(predictions - y_batch) / len(y_batch)

                self.theta -= lr * gradient_mse  # Обновление параметров

            loss_ = self.compute_loss(X_, y_)
            self.loss_history.append(loss_)

            if verbose and epoch_ % 20 == 0:
                print(f"Эпоха {epoch_}: Loss = {loss_:.4f}, LR = {lr:.6f}")

    def predict(self, X_):
        return X_.dot(self.theta)


def decay_exponential(epoch_, lr_initial, decay_rate=0.99):
    """Функция изменения learning rate по экспоненте"""
    return lr_initial * (decay_rate ** epoch_)


def decay_by_step(epoch_, lr_initial, drop_every=20, drop_factor=0.5):
    """Функция изменения learning rate по шагам"""
    return lr_initial * (drop_factor ** (epoch_ // drop_every))


# 3. Готовые оптимизаторы из библиотек
from sklearn.linear_model import SGDRegressor
from torch.optim import SGD, Adam, RMSprop
import torch
import torch.nn as nn


class PyTorchLinearRegression(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features,  out_features=1)

    def forward(self, x):
        return self.linear(x)


def evaluate_model(name, y_true, y_predicted_, time_trained_):
    """Для оценки моделей"""
    mse_ = mean_squared_error(y_true, y_predicted_)
    r2_ = r2_score(y_true, y_predicted_)
    print(f"{name:30} MSE: {mse_:.4f}, R²: {r2_:.4f}, Время: {time_trained_:.4f}с")
    return mse_, r2_

print('''

###########################################
#  ЭКСПЕРИМЕНТ 1: Различный размер батча  #
###########################################
''')

batch_sizes = [1, 8, 32, 128, X_train.shape[0]]  # Последний -- полный GD
results_batch = []

for batch_size in batch_sizes:

    model = LinearRegressionSGD(
        n_features=X_train.shape[1],
        learning_rate=0.01,
        batch_size_=batch_size
    )

    time_start = time.perf_counter()
    model.fit(X_train, y_train, n_epochs=100, verbose=False)
    time_trained = time.perf_counter() - time_start

    y_predicted = model.predict(X_test)
    y_predicted_original = scaler_y.inverse_transform(y_predicted)
    y_test_original = scaler_y.inverse_transform(y_test)

    mse, r2 = evaluate_model(
        name=f"Batch size {batch_size} "
             f"({'Полный GD' if batch_size == X_train.shape[0] else 'SGD'})",
        y_true=y_test_original, y_predicted_=y_predicted_original,
        time_trained_=time_trained
    )

    results_batch.append({
        'batch_size': batch_size,
        'mse': mse,
        'r2': r2,
        'time': time_trained,
        'loss_history': model.loss_history
    })

print('''

###########################################################
#  ЭКСПЕРИМЕНТ 2: Разные функции изменения learning rate  #
###########################################################
''')

lr_schedules = [
    ('Постоянный LR', None),
    ('Экспоненциальный затухание',
     lambda epoch_, lr: decay_exponential(epoch_, lr, decay_rate=0.99)),
    ('Ступенчатый',
     lambda epoch_, lr: decay_by_step(epoch_, lr, drop_every=25, drop_factor=0.5))
]

lr_results = []

for schedule_name, schedule_func in lr_schedules:

    model = LinearRegressionSGD(
        n_features=X_train.shape[1],
        learning_rate=0.1,  # Начальный LR увеличен для наглядности
        batch_size_=32
    )

    time_start = time.perf_counter()
    model.fit(
        X_train, y_train,
        n_epochs=100,
        learning_rate_schedule=schedule_func,
        verbose=False
    )
    time_trained = time.perf_counter() - time_start

    y_predicted = model.predict(X_test)
    y_predicted_original = scaler_y.inverse_transform(y_predicted)

    # noinspection PyUnboundLocalVariable
    mse, r2 = evaluate_model(
        schedule_name,
        y_test_original,
        y_predicted_original,
        time_trained
    )

    lr_results.append({
        'schedule': schedule_name,
        'mse': mse,
        'r2': r2,
        'time': time_trained,
        'loss_history': model.loss_history,
        'final_lr': schedule_func(99, 0.1) if schedule_func else 0.1
    })

print('''

######################################################
#  ЭКСПЕРИМЕНТ 3: Готовые оптимизаторы из библиотек  #
######################################################
''')

print("\n3.1. SGDRegressor из sklearn:")
sklearn_sgd = SGDRegressor(
    max_iter=100,
    tol=1e-3,
    learning_rate='constant',
    eta0=0.01,
    random_state=42
)

time_start = time.perf_counter()
# Убираем столбец единиц, так как SGDRegressor добавляет его сам
sklearn_sgd.fit(X_train[:, 1:], y_train.ravel())
time_sklearn = time.perf_counter() - time_start

y_predicted_sklearn = sklearn_sgd.predict(X_test[:, 1:]).reshape(-1, 1)
y_predicted_sklearn_original = scaler_y.inverse_transform(y_predicted_sklearn)

sklearn_mse, sklearn_r2 = evaluate_model(
    name="SGDRegressor (sklearn)",
    y_true=y_test_original, y_predicted_=y_predicted_sklearn_original,
    time_trained_=time_sklearn
)

print("\n3.2. PyTorch с разными оптимизаторами:")

# Подготовка данных для PyTorch
X_train_torch = torch.FloatTensor(X_train[:, 1:])  # Убираем столбец единиц
y_train_torch = torch.FloatTensor(y_train)
X_test_torch = torch.FloatTensor(X_test[:, 1:])
y_test_torch = torch.FloatTensor(y_test)

# Конфигурации оптимизаторов
# noinspection SpellCheckingInspection
optimizers_config = [
    ('SGD', SGD, {'lr': 0.01}),
    ('SGD с Momentum', SGD, {'lr': 0.01, 'momentum': 0.9}),
    ('SGD с Nesterov', SGD, {'lr': 0.01, 'momentum': 0.9, 'nesterov': True}),
    ('Adam', Adam, {'lr': 0.01}),
    ('RMSprop', RMSprop, {'lr': 0.01})
]

results_torch = []

for opt_name, opt_class, opt_params in optimizers_config:

    # Модель
    model = PyTorchLinearRegression(X_train.shape[1] - 1)
    criterion = nn.MSELoss()
    optimizer = opt_class(model.parameters(), **opt_params)

    # Обучение
    loss_history = []
    time_start = time.perf_counter()
    #
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(X_train_torch)
        loss = criterion(outputs, y_train_torch)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
    #
    time_trained = time.perf_counter() - time_start

    with torch.no_grad():  # Оценка
        y_predicted_torch = model(X_test_torch).numpy()

    y_predicted_torch_original = scaler_y.inverse_transform(y_predicted_torch)

    mse, r2 = evaluate_model(
        name=f"PyTorch {opt_name}",
        y_true=y_test_original,
        y_predicted_=y_predicted_torch_original,
        time_trained_=time_trained
    )

    results_torch.append({
        'optimizer': opt_name,
        'mse': mse,
        'r2': r2,
        'time': time_trained,
        'loss_history': loss_history
    })

#############################################################################################
#  ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ: Сравнение различных подходов к оптимизации линейной регрессии  #
#############################################################################################

# 1. Сравнение размеров батча
plt.figure(num="РзмБтч", figsize=(12, 7))
for result in results_batch:
    label = f'Batch {result["batch_size"]}' \
        if result["batch_size"] != X_train.shape[0] else 'Full GD'
    plt.plot(result['loss_history'][:50], label=label)
plt.xlabel('Эпоха', fontsize=12)
plt.ylabel('Loss (MSE)', fontsize=12)
plt.title('Влияние размера батча на сходимость', fontsize=14)
plt.legend()
plt.grid(visible=True, alpha=0.4)
plt.tight_layout()
plt.show(block=False)

# 2. Сравнение learning rate schedules
plt.figure(num="LeaRatSch", figsize=(12, 7))
for result in lr_results:
    plt.plot(result['loss_history'][:50], label=result['schedule'])
plt.xlabel('Эпоха', fontsize=12)
plt.ylabel('Loss (MSE)', fontsize=12)
plt.title('Разные стратегии изменения Learning Rate', fontsize=14)
plt.legend()
plt.grid(visible=True, alpha=0.4)
plt.tight_layout()
plt.show(block=False)

# 3. Сравнение готовых оптимизаторов
plt.figure(num="ГотОпт", figsize=(12, 7))
for result in results_torch[:3]:  # Только SGD варианты
    plt.plot(result['loss_history'][:50], label=result['optimizer'])
plt.xlabel('Эпоха', fontsize=12)
plt.ylabel('Loss (MSE)', fontsize=12)
plt.title('SGD и его модификации', fontsize=14)
plt.legend()
plt.grid(visible=True, alpha=0.4)
plt.tight_layout()
plt.show(block=False)

# 4. Сравнение современных оптимизаторов
plt.figure(num="СврОпт", figsize=(12, 7))
for result in results_torch[3:]:  # Adam и RMSprop
    plt.plot(result['loss_history'][:50], label=result['optimizer'])
plt.xlabel('Эпоха', fontsize=12)
plt.ylabel('Loss (MSE)', fontsize=12)
plt.title('Современные адаптивные оптимизаторы', fontsize=14)
plt.legend()
plt.tight_layout()
plt.grid(visible=True, alpha=0.4)

# Для пп. 5, 6
best_batch = min(results_batch, key=lambda x: x['mse'])
best_lr = min(lr_results, key=lambda x: x['mse'])
best_torch = min(results_torch, key=lambda x: x['mse'])
#
all_results = [
    ('Лучший Batch', best_batch['r2']),
    (best_lr['schedule'], best_lr['r2']),
    ('SGDRegressor', sklearn_r2),
    (f'PyTorch {best_torch["optimizer"]}', best_torch['r2'])
]
#
names = [r[0] for r in all_results]
r2_scores = [r[1] for r in all_results]

"""  # Схожие результаты, ненаглядно
# 5. Сравнение качества (R² score)
plt.figure(num="КачR2", figsize=(12, 8))
#
bars = plt.bar(names, r2_scores, color=['blue', 'green', 'orange', 'red'])
plt.xlabel('Метод', fontsize=12)
plt.ylabel('R² score', fontsize=12)
plt.title('Сравнение качества моделей (R²)', fontsize=14)
plt.ylim(0, 0.5)
plt.grid(visible=True, alpha=0.4, axis='y')
#
# Добавление значений на столбцы
for bar, score in zip(bars, r2_scores):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
            s=f'{score:.3f}', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.show(block=False)
"""

# 6. Сравнение времени обучения
plt.figure(num="ВреОбу", figsize=(12, 7))
times = [best_batch['time'], best_lr['time'], time_sklearn, best_torch['time']]
bars = plt.bar(names, times, color=['blue', 'green', 'orange', 'red'])
plt.xlabel('Метод', fontsize=12)
plt.ylabel('Время (секунды)', fontsize=12)
plt.title('Сравнение времени обучения', fontsize=14)
plt.grid(visible=True, alpha=0.4, axis='y')
#
# Добавление значений на столбцы
for bar, t in zip(bars, times):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height + 0.005,
            s=f'{t:.3f}s', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.show(block=True)

print('''

#################################
#  СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ  #
#################################
''')

summary_data = [
    [
        'Лучший Batch SGD',
        f'{best_batch["mse"]:.4f}',
        f'{best_batch["r2"]:.4f}',
        f'{best_batch["time"]:.4f}s'
    ],
    [
        best_lr['schedule'],
        f'{best_lr["mse"]:.4f}',
        f'{best_lr["r2"]:.4f}',
        f'{best_lr["time"]:.4f}s'
    ],
    [
        'SGDRegressor (sklearn)',
        f'{sklearn_mse:.4f}',
        f'{sklearn_r2:.4f}',
        f'{time_sklearn:.4f}s'
    ],
    [
        f'PyTorch {best_torch["optimizer"]}',
        f'{best_torch["mse"]:.4f}',
        f'{best_torch["r2"]:.4f}',
        f'{best_torch["time"]:.4f}s'
    ]
]

from tabulate import tabulate

headers = ['Метод', 'MSE', 'R²', 'Время обучения']
print(tabulate(summary_data, headers=headers, tablefmt='grid'))
