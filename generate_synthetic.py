"""
generate_synthetic.py

Generates a synthetic "employee churn" dataset with intentional, KNOWN
problems planted on purpose, deterministically (never left to random chance).
"""

import numpy as np
import pandas as pd

np.random.seed(42)
n_rows = 200

departments = np.random.choice(
    ["Engineering", "Sales", "Marketing", "HR"],
    size=n_rows,
    p=[0.37, 0.25, 0.2, 0.18],
)

age = np.random.randint(22, 60, size=n_rows).astype(float)
age[np.random.choice(n_rows, size=15, replace=False)] = np.nan

monthly_salary = np.random.normal(50000, 15000, size=n_rows).round(2)
monthly_salary[np.random.choice(n_rows, size=10, replace=False)] = np.nan

churned = np.random.choice(["Yes", "No"], size=n_rows, p=[0.2, 0.8])

exit_interview_completed = np.array(
    ["Yes" if (c == "Yes" and np.random.random() > 0.02) else "No" for c in churned]
)

last_bonus_pct = np.random.uniform(0, 15, size=n_rows).round(1).astype(object)
mixed_indices = np.random.choice(n_rows, size=8, replace=False)
for i in mixed_indices:
    last_bonus_pct[i] = f"{last_bonus_pct[i]}%"

years_of_experience = np.random.randint(0, 35, size=n_rows).astype(object)

df = pd.DataFrame({
    "employee_id": [f"EMP-{1000+i}" for i in range(n_rows)],
    "age": age,
    "department": departments,
    "monthly_salary": monthly_salary,
    "country": ["India"] * n_rows,
    "is_active": np.random.choice([True, False], size=n_rows, p=[0.98, 0.02]),
    "last_bonus_pct": last_bonus_pct,
    "years_of_experience": years_of_experience,
    "exit_interview_completed": exit_interview_completed,
    "churned": churned,
})

df["salary_copy"] = df["monthly_salary"]

df.loc[11] = df.loc[10]
df.loc[[20, 45], "department"] = "Enginering"
df.loc[[5, 15, 25], "years_of_experience"] = " "
df.loc[[35, 45], "years_of_experience"] = "?"

df.to_csv("data/sample/employee_churn_synthetic.csv", index=False)

print(df.head(15).to_string())
print("\nShape:", df.shape)
print("\nDtypes:\n", df.dtypes)