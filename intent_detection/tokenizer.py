class SacreMosesTokenizer(object):
    def __init__(self):
        try:
            from sacremoses import MosesTokenizer
            self._tokenizer = MosesTokenizer()
        except (ImportError, TypeError) as err:
            print('sacremoses is not installed. '
                  'To install sacremoses, use pip install -U sacremoses'
                  ' Now try NLTKMosesTokenizer using NLTK ...')
            raise

    def __call__(self, sample, return_str=False):
        """
        Parameters
        ----------
        sample: str
            The sentence to tokenize
        return_str: bool, default False
            True: return a single string
            False: return a list of tokens

        Returns
        -------
        ret : list of strs or str
            List of tokens or tokenized text
        """
        return self._tokenizer.tokenize(sample, return_str=return_str, escape=False)
