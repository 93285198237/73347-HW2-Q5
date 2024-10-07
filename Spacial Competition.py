import numpy as np
from scipy.optimize import fsolve


n_prices = 41

def expected_payoff_W(q_W, q_P, j):
    term1 = 1000 * 0.5 * j * np.sum(q_P[j+3:]) if j+3 < n_prices else 0
    term2 = 400 * 0.5 * j * np.sum(q_P[max(j-1, 0):min(j+2, n_prices)])
    term3 = 200 * 0.5 * j * q_P[j-2] if j-2 >= 0 else 0
    term4 = 700 * 0.5 * j * q_P[j+2] if j+2 < n_prices else 0
    term5 = 0.5 * j * np.sum(q_P[:max(j-3, 0)])
    return term1 + term2 + term3 + term4 + term5

def expected_payoff_P(q_W, q_P, j):
    term1 = 1000 * 0.5 * j * np.sum(q_W[j+3:]) if j+3 < n_prices else 0
    term2 = 600 * 0.5 * j * np.sum(q_W[max(j-1, 0):min(j+2, n_prices)])
    term3 = 300 * 0.5 * j * q_W[j-2] if j-2 >= 0 else 0
    term4 = 800 * 0.5 * j * q_W[j+2] if j+2 < n_prices else 0
    term5 = 0.5 * j * np.sum(q_W[:max(j-3, 0)])
    return term1 + term2 + term3 + term4 + term5


def equations(vars):
    q_W = vars[:n_prices]  
    q_P = vars[n_prices:]  
    
    eqs = []

    
    E_W_0 = expected_payoff_W(q_W, q_P, 0)
    for j in range(1, n_prices):
        eqs.append(expected_payoff_W(q_W, q_P, j) - E_W_0)
    
    
    E_P_0 = expected_payoff_P(q_W, q_P, 0)
    for j in range(1, n_prices):
        eqs.append(expected_payoff_P(q_W, q_P, j) - E_P_0)
    
    
    eqs.append(np.sum(q_W) - 1)
    eqs.append(np.sum(q_P) - 1)
    
    return eqs


initial_guess = np.ones(2 * n_prices) / n_prices


solution = fsolve(equations, initial_guess)


q_W_solution = np.clip(solution[:n_prices], 0, None)
q_P_solution = np.clip(solution[n_prices:], 0, None)


q_W_solution /= np.sum(q_W_solution)
q_P_solution /= np.sum(q_P_solution)


print("Wexford probabilities:", q_W_solution)
print("Pittsburgh probabilities:", q_P_solution)