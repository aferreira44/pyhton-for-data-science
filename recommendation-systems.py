import numpy as np
from lightfm import LightFM
from lightfm.datasets import fetch_movielens
from lightfm.evaluation import precision_at_k

# Recommendation Systems (Collaborative, Content-based)

# fetch data and format it
data = fetch_movielens(min_rating=4.0)

# print training and testing data
print(repr(data['train']))
print(repr(data['test']))

# create model
model = LightFM(loss='warp')

# train model
model.fit(data['train'], epochs=30, num_threads=2)

# Evaluate the trained model
test_precision = precision_at_k(model, data['test'], k=5).mean()

def sampleRecommendation(model, data, user_ids):
    # number of users and movies in training data
    n_users, n_items = data['train'].shape

    # generate recommendations for each user we input
    for user_id in user_ids:

        # movies they already like 
        known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]

        # movies our model predicts they will like
        scores = model.predict(user_id, np.arange(n_items))

        # rank them in order of most liked to least
        top_items = data['item_labels'][np.argsort(-scores)]

        # print out the results
        print("User %s" % user_id)
        print("     Know positives:")

        for x in known_positives[:3]:
            print("     %s" % x)

        print("     Recommended:")

        for x in top_items[:3]:
            print("     %s" % x)

sampleRecommendation(model, data, [3, 25, 450])