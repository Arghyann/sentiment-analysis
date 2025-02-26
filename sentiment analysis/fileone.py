import pandas as pd 
import matplotlib.pyplot as plt
df=pd.read_csv("IMDB Dataset.csv")
print(df.columns)
sentiment_counts = df.iloc[:,1].value_counts()

plt.figure(figsize=(6,4))
sentiment_counts.plot(kind="bar", color=["green", "red"])
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.title("Positive vs Negative Sentiment Distribution")
plt.xticks(rotation=0)  # Keep labels horizontal
plt.show()