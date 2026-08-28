from sklearn.naive_bayes import CategoricalNB

class ActivityRecognizer:
    def __init__(self, x, y):
        self.X = x
        self.y = y
        self.model = CategoricalNB()
        self.model.fit(self.X, self.y)

    def predict(self, test):
        prediction = self.model.predict(test)

        return prediction
