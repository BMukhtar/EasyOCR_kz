from label_studio_ml.model import LabelStudioMLBase


class DummyModel(LabelStudioMLBase):

    def __init__(self, **kwargs):
        # don't forget to call base class constructor
        super(DummyModel, self).__init__(**kwargs)
    
        # you can preinitialize variables with keys needed to extract info from tasks and annotations and form predictions
        from_name, schema = list(self.parsed_label_config.items())[0]
        self.from_name = from_name
        self.to_name = schema['to_name'][0]
        self.labels = schema['labels']

    def predict(self, tasks, **kwargs):
        """ This is where inference happens: model returns 
            the list of predictions based on input list of tasks 
        """
        predictions = []
        for task in tasks:
            predictions.append({
                'score': 0.987,  # prediction overall score, visible in the data manager columns
                'model_version': 'delorean-20151021',  # all predictions will be differentiated by model version
                'result': [{
                    'from_name': self.from_name,
                    'to_name': self.to_name,
                    'type': 'choices',
                    'score': 0.5,  # per-region score, visible in the editor 
                    'value': {
                        'choices': [self.labels[0]]
                    }
                }]
            })
        return predictions

    def fit(self, annotations, **kwargs):
        """ This is where training happens: train your model given list of annotations, 
            then returns dict with created links and resources
        """
        return {'path/to/created/model': 'my/model.bin'}