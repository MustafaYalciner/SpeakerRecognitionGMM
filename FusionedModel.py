

class FusionedModel:
    def __init__(self, foreground_model, background_model):
        self.foreground_model = foreground_model
        self.background_model = background_model


    def score_samples(self, development_set):
        scores = [None] * len(development_set)
        foreground_scores = self.foreground_model.score_samples(development_set)
        background_scores = self.background_model.score_samples(development_set)
        for i in range(len(scores)):
            scores[i] = foreground_scores[i]-background_scores[i]
        return scores
