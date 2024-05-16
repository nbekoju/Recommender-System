from surprise import dump
model_filename = "svd_model.pkl"
svd_collaborative_model = dump.load(model_filename)[1]
