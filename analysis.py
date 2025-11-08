import pandas as pd
import matplotlib.pyplot as plt
from google import genai  # Reintegrated Gemini API
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
import time


warnings.filterwarnings('ignore')

API_KEY = "MY_API"
client = genai.Client(api_key=API_KEY)


print("\n=== 1. LOADING DATA ===")

try:
    df = pd.read_csv("food_data.csv", encoding="utf-16", quotechar='"')
except FileNotFoundError:
    print("Error: food_data.csv not found. Make sure it's in the same directory.")
    exit()
except Exception as e:
    print(f"Error reading CSV: {e}")
    print("Please ensure your CSV is not open in another program.")
    exit()

print("   Data loaded successfully.")

df['date'] = pd.to_datetime(df['date'], errors='coerce')
df.dropna(subset=['date'], inplace=True)
df['day_of_week'] = df['date'].dt.day_name()
df['review_text'] = df['review_text'].fillna('')

df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
df['calories'] = pd.to_numeric(df['calories'], errors='coerce')

df = df[~df['transaction_id'].str.contains('img|2BETWEEN|2This|2WHAT|txn_1im', na=False)]


print("\n=== 2. ANALYZING DAY-WISE PATTERNS ===")
pattern = df.groupby(['day_of_week', 'item_category'])['quantity'].sum().unstack(fill_value=0)
days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
pattern = pattern.reindex(days_order)

print(pattern)
pattern.plot(kind='bar', figsize=(12, 6), stacked=True)
plt.title('Consumption Pattern by Day and Item Category')
plt.ylabel('Total Quantity Sold')
plt.xlabel('Day of the Week')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('pattern_chart.png')
print("   Saved 'pattern_chart.png'")

# ---------------------------
# 2. SENTIMENT ANALYSIS (Gemini API)

def analyze_sentiment(review):

    if not review.strip():
        return 'Neutral'
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=f"Classify the sentiment of this food review as Positive, Negative, or Neutral: {review}"
        )
        sentiment = response.text.strip()
        if sentiment not in ['Positive', 'Negative', 'Neutral']:
            sentiment = 'Neutral'
            time.sleep(4)

        return sentiment

    except Exception as e:
        print(f"Error analyzing sentiment: {e}")

        return 'Neutral'

print("\n=== RUNNING SENTIMENT ANALYSIS ON ALL REVIEWS ===")
df['sentiment'] = df['review_text'].apply(analyze_sentiment)

df.to_csv("sentiment_results_temp.csv", index=False)
print("Saved intermediate sentiment results -> sentiment_results_temp.csv")

# 3. AGGREGATE SENTIMENT BY ITEM
# ===============================
print("\n=== AGGREGATING SENTIMENT BY FOOD ITEM ===")

# Count how many Positive, Negative, and Neutral per item
sentiment_counts = (
    df.groupby(['item_name', 'sentiment'])
    .size()
    .unstack(fill_value=0)
    .reset_index()
)

for col in ['Positive', 'Negative', 'Neutral']:
    if col not in sentiment_counts.columns:
        sentiment_counts[col] = 0

# Compute final sentiment for each item
def decide_final_sentiment(row):
    pos, neg, neu = row['Positive'], row['Negative'], row['Neutral']
    if pos > neg:
        return 'Positive'
    elif neg > pos:
        return 'Negative'
    else:
        return 'Neutral'  # Tie or only Neutral reviews

sentiment_counts['final_sentiment'] = sentiment_counts.apply(decide_final_sentiment, axis=1)

print("\n=== FINAL SENTIMENT RESULTS ===")
print(sentiment_counts[['item_name', 'Positive', 'Negative', 'Neutral', 'final_sentiment']])

# Save results
sentiment_counts.to_csv("final_item_sentiments.csv", index=False)
print("\nSaved final aggregated sentiment results -> final_item_sentiments.csv")

# ----------------------------
# 4. DEMAND PREDICTION MODEL (RANDOM FOREST)
# ----------------------------
from sklearn.ensemble import RandomForestRegressor
import numpy as np

print("\n=== 4. TRAINING DEMAND PREDICTION MODEL (Random Forest) ===")

# Map sentiment to numeric scores
df['sentiment_score'] = df['sentiment'].map({'Positive': 1, 'Neutral': 0, 'Negative': -1}).fillna(0)

# Prepare data for the model
df_model = pd.get_dummies(
    df,
    columns=['day_of_week', 'item_category', 'item_type', 'student_diet_pref', 'time_of_day'],
    drop_first=True
)

numeric_features = ['price', 'calories', 'rating', 'sentiment_score']
df_model[numeric_features] = df_model[numeric_features].fillna(0)

dummy_cols = [
    col for col in df_model.columns
    if col.startswith(('day_of_week_', 'item_category_', 'item_type_', 'student_diet_pref_', 'time_of_day_'))
]
features = numeric_features + dummy_cols

X = df_model[features]
y = df_model['quantity'].fillna(1)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize Random Forest
model = RandomForestRegressor(
    n_estimators=200,
    random_state=1,
    max_depth=10,
    min_samples_split=4,
    n_jobs=-1
)

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"   Mean Absolute Error: {mae:.2f}")
print(f"   R^2 Score: {r2:.2f}")

# Plot prediction performance
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel('Actual Quantity Sold')
plt.ylabel('Predicted Quantity Sold')
plt.title('Demand Prediction Accuracy (Random Forest)')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.tight_layout()
plt.savefig('demand_prediction_rf.png')
print("   Saved 'demand_prediction_rf.png'")

# ----------------------------
# 5. FORECASTING TOP DEMAND PER DAY
# ----------------------------
print("\n=== 5. FORECASTING TOP DEMAND PER DAY ===")

# Get unique items and average numeric values
unique_items = df[['item_name', 'item_category', 'item_type', 'student_diet_pref']].drop_duplicates()

# Average values to use for simulation
avg_price = df['price'].mean()
avg_calories = df['calories'].mean()
avg_rating = df['rating'].mean()

# Days and time slots for simulation
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
times = ['Breakfast', 'Lunch', 'Dinner']

# Create synthetic data (all combinations)
simulated_data = []
for _, item in unique_items.iterrows():
    for day in days:
        for time in times:
            simulated_data.append({
                'item_name': item['item_name'],
                'item_category': item['item_category'],
                'item_type': item['item_type'],
                'student_diet_pref': item['student_diet_pref'],
                'day_of_week': day,
                'time_of_day': time,
                'price': avg_price,
                'calories': avg_calories,
                'rating': avg_rating,
                'sentiment_score': 1  # assume positive sentiment baseline
            })

forecast_df = pd.DataFrame(simulated_data)

# Encode like training data
forecast_model = pd.get_dummies(
    forecast_df,
    columns=['day_of_week', 'item_category', 'item_type', 'student_diet_pref', 'time_of_day'],
    drop_first=True
)

# Align columns with trained model
for col in X.columns:
    if col not in forecast_model.columns:
        forecast_model[col] = 0
forecast_model = forecast_model[X.columns]

# Predict demand
forecast_df['predicted_quantity'] = model.predict(forecast_model)

# Save predictions
forecast_df.to_csv("predicted_demand_by_day.csv", index=False)
print("Saved 'predicted_demand_by_day.csv'")

# ----------------------------
# 6. IDENTIFY TOP ITEMS PER DAY
# ----------------------------
print("\n=== TOP DEMANDED ITEMS BY DAY ===")
top_items = []

for day in days:
    day_df = forecast_df[forecast_df['day_of_week'] == day].sort_values(by='predicted_quantity', ascending=False)
    print(f"\n{day}:")
    for i, row in day_df.head(5).iterrows():
        print(f"  {row['item_name']} — Predicted {row['predicted_quantity']:.1f} servings")
        top_items.append({
            'day_of_week': day,
            'rank': len(top_items) % 5 + 1,
            'item_name': row['item_name'],
            'predicted_quantity': row['predicted_quantity']
        })
    top_items_df = pd.DataFrame(top_items)

# ----------------------------
# 7. VISUALIZE TOP 5 ITEMS BY DAY (Separate Subplots)
# ----------------------------
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()

for i, day in enumerate(days):
    ax = axes[i]
    day_df = top_items_df[top_items_df['day_of_week'] == day]
    ax.bar(day_df['item_name'], day_df['predicted_quantity'], color='skyblue')
    ax.set_title(day)
    ax.set_xticklabels(day_df['item_name'], rotation=75)
    ax.set_ylabel("Predicted Quantity")
    ax.set_xlabel("Item")

plt.suptitle("Predicted Top 5 Food Items per Day")
plt.tight_layout()
plt.savefig("top_items_by_day_subplots.png", dpi=300)
print("Saved 'top_items_by_day_subplots.png'")

# ----------------------------
# 8. SIMPLE RECOMMENDATION SYSTEM
# ----------------------------
print("\n=== 5. GENERATING SIMPLE RECOMMENDATIONS ===")

avg_ratings = df.groupby('item_name')['rating'].mean().sort_values(ascending=False)
print("\n--- Top 5 Rated Items (Overall) ---")
print(avg_ratings.head())

ratings_count = df.groupby('item_name')['rating'].count()
items_with_many_ratings = ratings_count[ratings_count > 2].index
best_rated_popular = avg_ratings.loc[items_with_many_ratings].sort_values(ascending=False)
print("\n--- Top Rated Items (with >2 ratings) ---")
print(best_rated_popular.head())


# ----------------------------
# 9. EXPORT DATA FOR FRONTEND
# ----------------------------
print("\n=== 7. EXPORTING DATA FOR FRONTEND ===")

# 1. Export sentiment summary
sentiment_counts.to_json("frontend_sentiments.json", orient="records")

# 2. Export day-wise forecast
forecast_df.to_json("frontend_predicted_demand.json", orient="records")

# 3. Export top items per day
top_items_df.to_json("frontend_top_items.json", orient="records")


print("Saved JSON files for frontend:")
print(" - frontend_sentiments.json")
print(" - frontend_predicted_demand.json")
print(" - frontend_top_items.json")
print(" - frontend_feature_importance.json")


# ----------------------------
# 10. FINAL INSIGHTS
# ----------------------------
print("\n=== 6. KEY INSIGHTS ===")
print("1. Day-wise and category-based consumption reveals peak days and preferences (see 'pattern_chart.png').")
print("2. Sentiment analysis identifies satisfaction and problem dishes (see 'sentiment_chart.png').")
print("3. Demand prediction prototype is built (see 'demand_prediction.png').")
print(f"4. Top rated popular item: {best_rated_popular.index[0]} (Rating: {best_rated_popular.iloc[0]:.2f})")

print("\n=== SCRIPT FINISHED ===\n")
