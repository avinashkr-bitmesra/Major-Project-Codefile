import numpy as np
import pandas as pd
import glob
import cvxpy as cp
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import json
import random

csv_files = glob.glob("hourly_5_Dec_*.csv")
N = len(csv_files)
T = 24

forecast = np.zeros((N, T))
for i, file in enumerate(csv_files):
    df = pd.read_csv(file)
    forecast[i] = df['load'].values[:T]

def generate_dynamic_tou_prices(forecast, daily_price=0.1644, base_spread=0.2):
    total_demand = forecast.sum(axis=0)

    norm_demand = (total_demand - total_demand.min()) / (total_demand.max() - total_demand.min())

    demand_std = np.std(total_demand)
    demand_mean = np.mean(total_demand)
    coeff_var = demand_std / demand_mean if demand_mean != 0 else 0

    spread_factor = base_spread + coeff_var
    peak_multiplier = 1 + spread_factor
    valley_multiplier = max(1 - spread_factor, 0.1)  

    price_floor = daily_price * valley_multiplier
    price_ceiling = daily_price * peak_multiplier
    tou_price = price_floor + (price_ceiling - price_floor) * norm_demand
    tou_price = gaussian_filter1d(tou_price, sigma=1.0)

    return tou_price



def optimize_individual_user_cvx(demand, price, lambda_, alpha=0.0, ramp_limit=None):
    x = cp.Variable(T)
    original_cost = float(np.dot(price, demand))

    obj = cp.sum(cp.multiply(price, x)) + lambda_ * cp.sum_squares(x - demand)
    if alpha > 0:
        obj += alpha * cp.sum_squares(x[1:] - x[:-1])
    objective = cp.Minimize(obj)

    constraints = [
        cp.sum(x) == cp.sum(demand),
        x >= 0,
        cp.sum(cp.multiply(price, x)) <= original_cost
    ]
    lower_bound = 0.5 * demand
    upper_bound = 1.5 * demand
    constraints += [x >= lower_bound, x <= upper_bound]

    if ramp_limit is not None:
        constraints += [cp.abs(x[1:] - x[:-1]) <= ramp_limit]
        
    prob = cp.Problem(objective, constraints)
    prob.solve()

    if prob.status != cp.OPTIMAL:
        print(f"Warning: Optimization failed with status {prob.status}")
        return demand

    return x.value

def generate_user_id():
    return ''.join([str(random.randint(0, 9)) for _ in range(12)])

lambda_vals = [0.001, 0.01, 0.1]
alpha_vals = [0.0, 0.1, 1, 10]

best_score = -np.inf
best_tradeoff_params = None

max_savings = -np.inf
best_savings_params = None

min_roughness = np.inf
best_smooth_params = None

results = []

p_t = generate_dynamic_tou_prices(forecast)

for lambda_ in lambda_vals:
    for alpha in alpha_vals:
        x_grid = np.zeros_like(forecast)
        total_savings = 0.0
        total_roughness = 0.0

        for i in range(N):
            optimized = optimize_individual_user_cvx(forecast[i], p_t, lambda_, alpha)
            x_grid[i] = optimized

            orig_cost = np.dot(p_t, forecast[i])
            opt_cost = np.dot(p_t, optimized)
            savings = orig_cost - opt_cost
            roughness = np.sum((optimized[1:] - optimized[:-1]) ** 2)

            total_savings += savings
            total_roughness += roughness

        score = total_savings - 0.1 * total_roughness  

        results.append({
            "lambda": lambda_,
            "alpha": alpha,
            "score": score,
            "total_savings": total_savings,
            "total_roughness": total_roughness
        })

        if score > best_score:
            best_score = score
            best_tradeoff_params = (lambda_, alpha)
            best_x_tradeoff = x_grid.copy()

        if total_savings > max_savings:
            max_savings = total_savings
            best_savings_params = (lambda_, alpha)
            best_x_savings = x_grid.copy()

        if total_roughness < min_roughness:
            min_roughness = total_roughness
            best_smooth_params = (lambda_, alpha)
            best_x_smooth = x_grid.copy()

pd.DataFrame(results).to_csv("parameter_grid_results.csv", index=False)

print("\n🏆 Best Overall Trade-Off:")
print("Lambda:", best_tradeoff_params[0], "Alpha:", best_tradeoff_params[1])

print("\n💰 Best Cost Savings:")
print("Lambda:", best_savings_params[0], "Alpha:", best_savings_params[1], f"(${max_savings:.2f})")

print("\n🌊 Best Smoothness (Lowest Roughness):")
print("Lambda:", best_smooth_params[0], "Alpha:", best_smooth_params[1], f"(Roughness: {min_roughness:.2f})")

x_new = best_x_tradeoff

# ---------------------- JSON EXPORT -----------------------

random.seed(42)
all_users_data = []

for i in range(N):
    original_load = forecast[i]
    optimized_load = x_new[i]


    original_hourly_cost = original_load * p_t
    optimized_hourly_cost = optimized_load * p_t
    hourly_savings = original_hourly_cost - optimized_hourly_cost

    total_original_cost = original_hourly_cost.sum()
    total_optimized_cost = optimized_hourly_cost.sum()
    total_savings = total_original_cost - total_optimized_cost
    savings_percent = (total_savings / total_original_cost) * 100

    user_id = generate_user_id()

    user_data = {
        "User ID": user_id,
        "Hourly Data": [
            {
                "Hour": hour + 1,
                "Forecasted Load (kWh)": float(original_load[hour]),
                "Optimized Load (kWh)": float(optimized_load[hour]),
                "Original Cost ($)": float(original_hourly_cost[hour]),
                "Optimized Cost ($)": float(optimized_hourly_cost[hour]),
                "Hourly Savings ($)": float(hourly_savings[hour])
            }
            for hour in range(T)
        ],
        "Summary": {
            "Total Original Cost ($)": float(total_original_cost),
            "Total Optimized Cost ($)": float(total_optimized_cost),
            "Total Savings ($)": float(total_savings),
            "Savings Percent (%)": float(savings_percent)
        }
    }

    all_users_data.append(user_data)

with open("users_comparison.json", "w") as f:
    json.dump(all_users_data, f, indent=4)

print("\n========== PER-USER SAVINGS FOR EACH (λ, α) COMBINATION ==========")

for lambda_ in lambda_vals:
    for alpha in alpha_vals:
        print(f"\n--- Lambda: {lambda_}, Alpha: {alpha} ---")
        for i in range(N):
            optimized = optimize_individual_user_cvx(forecast[i], p_t, lambda_, alpha)
            original_cost = np.dot(p_t, forecast[i])
            optimized_cost = np.dot(p_t, optimized)
            savings = original_cost - optimized_cost
            savings_percent = (savings / original_cost) * 100 if original_cost > 0 else 0.0

            print(f"User {i+1:>2}: Original = ${original_cost:.2f}, "
                  f"Optimized = ${optimized_cost:.2f}, "
                  f"Savings = ${savings:.2f} ({savings_percent:.2f}%)")


user_savings_records = []

for lambda_ in lambda_vals:
    for alpha in alpha_vals:
        for i in range(N):
            optimized = optimize_individual_user_cvx(forecast[i], p_t, lambda_, alpha)
            original_cost = np.dot(p_t, forecast[i])
            optimized_cost = np.dot(p_t, optimized)
            savings = original_cost - optimized_cost
            savings_percent = (savings / original_cost) * 100 if original_cost > 0 else 0.0

            user_savings_records.append({
                "User": i+1,
                "Lambda": lambda_,
                "Alpha": alpha,
                "Original Cost ($)": original_cost,
                "Optimized Cost ($)": optimized_cost,
                "Savings ($)": savings,
                "Savings Percent (%)": savings_percent
            })

pd.DataFrame(user_savings_records).to_csv("user_savings_by_lambda_alpha.csv", index=False)


for i, lambda_ in enumerate(lambda_vals):
    for j, alpha in enumerate(alpha_vals):
        x_grid = np.zeros_like(forecast)
        for u in range(N):
            x_grid[u] = optimize_individual_user_cvx(forecast[u], p_t, lambda_, alpha)

        total_opt_load = x_grid.sum(axis=0)
        total_orig_load = forecast.sum(axis=0)

        fig, ax = plt.subplots(figsize=(6, 4))
        twin_ax = ax.twinx()

        ax.plot(total_orig_load, label="Forecasted Load", color="blue")
        ax.plot(total_opt_load, label="Optimized Load", color="green", linestyle='--')

        twin_ax.plot(p_t, color="red", linestyle=':', label="TOU Price")
        twin_ax.set_ylabel("Price ($/kWh)", color="red", fontsize=9)
        twin_ax.tick_params(axis='y', labelcolor="red", labelsize=8)
        twin_ax.set_ylim(p_t.min() * 0.9, p_t.max() * 1.1)

        ax.set_title(f"λ={lambda_}, α={alpha}", fontsize=10)
        ax.set_xlabel("Hour", fontsize=9)
        ax.set_ylabel("Load (kWh)", fontsize=9)
        ax.tick_params(axis='both', labelsize=8)
        ax.grid(True)

        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = twin_ax.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=8)

        plt.tight_layout()
        plt.savefig(f"plot_{i}_{j}.png", dpi=300)
        plt.close()

# ---------------------- PLOT WITHOUT OPTIMIZED LOAD -----------------------

total_orig_load = forecast.sum(axis=0)

fig, ax = plt.subplots(figsize=(10, 5))
twin_ax = ax.twinx()

ax.plot(total_orig_load, label="Forecasted Load", color="blue")
ax.set_xlabel("Hour", fontsize=10)
ax.set_ylabel("Load (kWh)", fontsize=10, color="blue")
ax.tick_params(axis='y', labelcolor="blue", labelsize=9)
ax.set_title("Forecasted Load vs TOU Price (No Optimization)", fontsize=12)
ax.grid(True)

twin_ax.plot(p_t, color="red", linestyle=':', label="TOU Price")
twin_ax.set_ylabel("Price ($/kWh)", color="red", fontsize=10)
twin_ax.tick_params(axis='y', labelcolor="red", labelsize=9)
twin_ax.set_ylim(p_t.min() * 0.9, p_t.max() * 1.1)

h1, l1 = ax.get_legend_handles_labels()
h2, l2 = twin_ax.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=9)

plt.tight_layout()
plt.savefig("forecasted_load_vs_price.png", dpi=300)
plt.show()
