import matplotlib.pyplot as plt
import seaborn as sns
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# Load the MovieLens 100K dataset
reader = Reader(line_format='user item rating timestamp', sep='\t')
data = Dataset.load_from_file('data/u.data', reader=reader)

# Split the data into training and testing sets
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Create an SVD (Singular Value Decomposition) model
model = SVD()

# Train the model on the training set
model.fit(trainset)

# Make predictions on the test set
predictions = model.test(testset)

# Calculate RMSE (Root Mean Squared Error) to evaluate the model
rmse = accuracy.rmse(predictions)
print(f"RMSE: {rmse}")

# Create a histogram of user ratings
user_ratings = [pred.r_ui for pred in predictions]
plt.figure(figsize=(8, 6))
sns.histplot(user_ratings, kde=True, bins=20)
plt.xlabel("User Ratings")
plt.ylabel("Frequency")
plt.title("Distribution of User Ratings")
plt.show()

# Create a bar chart of top N recommended movies
user_id = '42'
user_ratings = []
for movie_id in range(1, 1683):
    user_ratings.append((movie_id, model.predict(user_id, movie_id).est))
user_ratings.sort(key=lambda x: x[1], reverse=True)
top_n = 10
top_movies = user_ratings[:top_n]
movie_ids, predicted_ratings = zip(*top_movies)
plt.figure(figsize=(10, 6))
sns.barplot(x=predicted_ratings, y=[f"Movie {id}" for id in movie_ids])
plt.xlabel("Predicted Ratings")
plt.ylabel("Movies")
plt.title(f"Top {top_n} Recommended Movies for User {user_id}")
plt.show()
