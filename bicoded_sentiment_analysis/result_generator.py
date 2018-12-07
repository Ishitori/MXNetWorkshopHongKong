class ResultFileGenerator:
    def __init__(self):
        self._record_template = '''
<Tweet id="{}">
	<Happiness>
	{}
	</Happiness>
	<Sadness>
	{}
	</Sadness>
	<Anger>
	{}
	</Anger>
	<Fear>
	{}
	</Fear>
	<Surprise>
	{}
	</Surprise>
	<Content>
	{}
	</Content>
</Tweet>        

        '''

    def write_results(self, predictions, file_path):
        sorted_predictions = sorted(predictions, key=lambda k: k['ri'])

        with open(file_path, mode='w', encoding='utf-8') as f:
            for prediction in sorted_predictions:
                f.write(self._record_template.format(
                    prediction['ri'],
                    self._get_string_bool(prediction['happiness']),
                    self._get_string_bool(prediction['sadness']),
                    self._get_string_bool(prediction['anger']),
                    self._get_string_bool(prediction['fear']),
                    self._get_string_bool(prediction['surprise']),
                    # generate fake, but unique content for official evaluation to work
                    prediction['ri']
                ))

    def _get_string_bool(self, value):
        return 'T' if value else 'F'
