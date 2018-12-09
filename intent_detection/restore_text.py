from tokenizer import SacreMosesTokenizer


def get_text_result(idxs, original_text, original_tokens):
    """Converts tokens to text

    Parameters
    ----------
    idx : `int`
        Question index
    answer_span : `Tuple`
        Answer span (start_index, end_index)

    Returns
    -------
    text : `str`
        A chunk of text for provided answer_span or None if answer span cannot be provided
    """

    indices = get_char_indices(original_text, original_tokens)

    start = idxs[0]
    end = idxs[len(idxs) - 1]
    text = original_text[indices[start][0]:indices[end][1]]
    return text


def get_char_indices(text, text_tokens):
    """Match token with character indices

    Parameters
    ----------
    text: str
        Text
    text_tokens: list[str]
        Tokens of the text

    Returns
    -------
    char_indices_per_token: List[Tuple]
        List of (start_index, end_index) of characters where the position equals to token index
    """
    char_indices_per_token = []
    current_index = 0

    for token in text_tokens:
        current_index = text.find(token, current_index)
        char_indices_per_token.append((current_index, current_index + len(token)))
        current_index += len(token)

    return char_indices_per_token
