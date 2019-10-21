#!/usr/bin/env python3

from collections import namedtuple


Result = namedtuple('Result', ('nfev', 'cost', 'gradnorm', 'x'))
Result.__doc__ = """Результаты оптимизации

Attributes
----------
nfev : int
    Полное число вызовов модельной функции
cost : 1-d array
    Значения функции потерь 0.5 sum(y - f)^2 на каждом итерационном шаге.
    В случае метода Гаусса—Ньютона длина массива равна nfev, в случае ЛМ-метода
    длина массива — менее nfev
gradnorm : float
    Норма градиента на финальном итерационном шаге
x : 1-d array
    Финальное значение вектора, минимизирующего функцию потерь
"""
import numpy as np
from collections import namedtuple

Result = namedtuple('Result', ('nfev', 'cost', 'gradnorm', 'x'))

def gauss_newton(y, f, j, x0, k=1, tol=1e-4):
    x = np.array(x0, dtype=np.float)
    cost = []
    i = 0
    while True:
        i += 1
        jac = j(*x)
        A = np.dot(jac.T, jac)
        r = y - f(*x)
        cost.append(0.5*np.dot(r, r))
        b = np.dot(jac.T, r)
        delta_x = np.linalg.solve(A, b)
        x += k * delta_x
        if np.linalg.norm(delta_x) <= tol:
            break
    gradnorm = np.linalg.norm(np.dot(jac.T, r))
    cost = np.array(cost)
    return Result(nfev=i, cost=cost, gradnorm=gradnorm, x=x)


def lm(y, f, j, x0, lmbd0=1e-2, nu=2, tol=1e-4):
    x = np.array(x0, dtype=np.float)
    lmbd = lmbd0
    cost = []
    i = 0
    delta_x = 1e+10
    while True:
        i += 2
        jac = j(*x)
        A0 = np.dot(jac.T, jac) + lmbd * np.identity(jac[1, :].size)
        Anu = np.dot(jac.T, jac) + lmbd/nu * np.identity(jac[1, :].size)
        r = y - f(*x)
        cost.append(0.5 * np.dot(r, r))
        b = np.dot(jac.T, r)
        delta_x0 = np.linalg.solve(A0, b)
        delta_xnu = np.linalg.solve(Anu, b)
        F = lambda t: 0.5*np.linalg.norm(y - f(*(x + t)))**2
        if F(delta_xnu) <= 0.5 * np.dot(r, r):
            lmbd = lmbd/nu
            delta_x = delta_xnu
            x += delta_x
        elif F(delta_xnu) <= F(delta_x0):
            delta_x = delta_x0
            x += delta_x
        else:
            lmbd = nu*lmbd
        if np.linalg.norm(delta_x) <= tol:
            break
    gradnorm = np.linalg.norm(np.dot(jac.T, r))
    cost = np.array(cost)
    return Result(nfev=i, cost=cost, gradnorm=gradnorm, x=x)
