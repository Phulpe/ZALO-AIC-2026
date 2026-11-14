class BlindPredictionError(ValueError):
    """Keep blind generation in the private ledger, never in exception messages."""
    def __init__(self,prediction_text):
        if not isinstance(prediction_text,str):raise TypeError('Prediction text required')
        super().__init__('Generated response is not a valid plan; retained privately')
        self.prediction_text=prediction_text
