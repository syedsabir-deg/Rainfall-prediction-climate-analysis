
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Data/Processed/rainfall_combined.csv")

monthly_rainfall = df.groupby("month")["rainfall"].mean()

monthly_rainfall.plot(kind="bar")

plt.title("Average Monthly Rainfall Across 724 Districts")
plt.xlabel("Month")
plt.ylabel("Average Rainfall")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()








