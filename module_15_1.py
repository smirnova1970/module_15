"""Домашнее задание: Знакомство с Google Colab
Цель задания:
Научиться основным функциям и возможностям Google Colab, включая выполнение кода, работу с текстовыми ячейками и загрузку данных.

Задание:
Создание и настройка документа:

Создайте новый документ в Google Colab.
Переименуйте документ в "Введение в Google Colab".
Текстовые и кодовые ячейки:

Вставьте текстовую ячейку и напишите краткое введение о Google Colab.
Вставьте новую кодовую ячейку и выполните следующий код, чтобы убедиться, что среда работает:
print("Hello, Google Colab!")
Основы Python:

Вставьте кодовую ячейку и напишите код для выполнения следующих задач:
Создайте список из 10 случайных чисел.
Напишите функцию, которая возвращает сумму чисел в списке.
Вызовите функцию и выведите результат на экран.
Работа с библиотеками:

Установите библиотеку numpy (если она не установлена):
!pip install numpy
Импортируйте библиотеку numpy и создайте массив из 20 случайных чисел.
Вычислите среднее значение массива и выведите его на экран.
Загрузка данных:

Загрузите любой CSV файл из интернета в ваш Google Colab.
Используйте библиотеку pandas для чтения CSV файла и выведите первые 5 строк данных.
Визуализация данных:

Установите библиотеку matplotlib (если она не установлена):
!pip install matplotlib
Постройте простой график на основе данных из вашего CSV файла.
Использование генетических алгоритмов:

Скопируйте код генетического алгоритма из лекционного блокнота.
Запустите его, изучите работу кода.
Измените значение целевой переменной TARGET. Добейтесь (увеличивая при необходимости число итераций обучения) полного соответствия "нового индивида целевой переменной".
Сохранение и совместное использование:

Сохраните ваш документ на Google Диске.
Настройте доступ к документу для общего использования и отправьте ссылку вашему преподавателю.
Критерии оценки:
Выполнение всех шагов задания.
Корректность и чистота кода.
Понятные и информативные текстовые ячейки.
Качество визуализации данных.
Дополнительные ресурсы:
Документация Google Colab: https://colab.research.google.com/notebooks/intro.ipynb
Документация NumPy: https://numpy.org/doc/
Документация pandas: https://pandas.pydata.org/pandas-docs/stable/
Документация Matplotlib: https://matplotlib.org/stable/users/index.html"""

# print("Hello, Google Colab!")
#
# from random import randint
#
# numbers = []
# for i in range(10):
#     numbers.append(randint(-10, 10))
#     sum = 0
# for j in numbers:
#     sum += j
# print(sum)
#
# import numpy as np
#
# !pip install numpy
#
#
# import numpy as np
# a = np.random.randint(1, 100, size = (20))
# print(a)
# suma = 0
# for i in range(20):
#   suma += a[i]
# print(suma / 20)
#
#
# !pip install pandas
#
# import pandas as pd
#
# url = 'https://people.sc.fsu.edu/~jburkardt/data/csv/airtravel.csv'
# data = pd.read_csv(url)
#
# print(data.head())
#
# # Загрузка CSV файла из интернета
# url = 'https://people.sc.fsu.edu/~jburkardt/data/csv/airtravel.csv'
# data = pd.read_csv(url)
#
# # Вывод первых 5 строк данных
# print(data.head())
#
# !pip install matplotlib
#
#
# import matplotlib.pyplot as plt
#
# plt.figure(figsize=(10, 6))
# plt.plot(data['Month'], data[' "1958"'], marker='o', linestyle='-')
# plt.title('Количество пассажиров по месяцам в 1958 году')
# plt.xlabel('Месяц')
# plt.ylabel('Количество пассажиров')
# plt.grid(True)
# plt.show()


import random

# Количество особей в каждом поколении
POPULATION_SIZE = 100

# Валидные гены
GENES = '''abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOP
QRSTUVWXYZ 1234567890, .-;:_!"#%&/()=?@${[]}'''

# Целевая строка для генерации
TARGET = "Imo, in principio erat casus 2024"
"""Класс, представляющий отдельную особь (индивида) в популяции"""
class Individual(object):
	def __init__(self, chromosome):
		self.chromosome = chromosome
		self.fitness = self.cal_fitness()

	@classmethod
	def mutated_genes(self):
		"""Создаем случайные гены для мутации"""
		global GENES
		gene = random.choice(GENES)
		return gene

	@classmethod
	def create_gnome(self):
		"""Создаем хромосому или набор генов"""
		global TARGET
		gnome_len = len(TARGET)
		return [self.mutated_genes() for _ in range(gnome_len)]

	def gene_transfer(self, par2):
		"""Передаем гены новому поколению индивидов"""
		child_chromosome = []
		for gp1, gp2 in zip(self.chromosome, par2.chromosome):
			prob = random.random()
			# если вероятность меньше 0,45, берем ген
			# от родителя 1
			if prob < 0.45:
				child_chromosome.append(gp1)
			# если вероятность между 0.45 и 0.90, берем
			# ген от родителя 2
			elif prob < 0.90:
				child_chromosome.append(gp2)
			# в противном случае берем случайный ген (мутация),
			else:
				child_chromosome.append(self.mutated_genes())
		return Individual(child_chromosome)

	def cal_fitness(self):
		"""Рассчитываем показатель соответствия, это количество символов в строке, которые отличаются от целевой
		строки."""
		global TARGET
		fitness = 0
		for gs, gt in zip(self.chromosome, TARGET):
			if gs != gt: fitness+= 1
		return fitness

# Driver code
def main():
	global POPULATION_SIZE
	#Текущее поколение
	generation = 1
	found = False
	population = []
	# Новое поколение
	for _ in range(POPULATION_SIZE):
				gnome = Individual.create_gnome()
				population.append(Individual(gnome))

	while not found:
		# Отсортируем популяцию в порядке возрастания оценки соответствия целевой функции
		population = sorted(population, key = lambda x:x.fitness)
		# Если у нас появился индивид, достигший целевой функции
		# цикл совершенствования можно прервать
		if population[0].fitness <= 0:
			found = True
			break
		# В противном случае - продолжаем создавать новые поколения
		new_generation = []
		# Определяем 10% популяции, наиболее соответствующих целевой фукнции
		# чтобы передать их гены будущим поколениям
		s = int((10*POPULATION_SIZE)/100)
		new_generation.extend(population[:s])
		s = int((90*POPULATION_SIZE)/100)
		for _ in range(s):
			parent1 = random.choice(population[:50])
			parent2 = random.choice(population[:50])
			child = parent1.gene_transfer(parent2)
			new_generation.append(child)
		population = new_generation
		print("Generation: {}\tString: {}\tFitness: {}".
			format(generation,
			"".join(population[0].chromosome),
			population[0].fitness))
		generation += 1
	print("Generation: {}\tString: {}\tFitness: {}".
		format(generation,
		"".join(population[0].chromosome),
		population[0].fitness))

if __name__ == '__main__':
	main()

