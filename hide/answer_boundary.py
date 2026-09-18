"""Explicit first-nonempty-answer-line stopping; the historical protocol stays available."""
import re


def first_answer_line(text):
    """Return (answer, boundary found, text after boundary), skipping leading whitespace."""
    start = re.search(r'\S', text)
    if start is None:
        return '', False, ''
    end = re.search(r'[\r\n]', text[start.start():])
    if end is None:
        return text[start.start():].strip(), False, ''
    position = start.start() + end.start()
    return text[start.start():position].strip(), True, text[position:].lstrip('\r\n')


def stopping_criteria(tokenizer, prompt_length):
    # A double newline need not use the standalone newline token ID.
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList

    class FirstAnswerLine(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            stopped = [first_answer_line(tokenizer.decode(row[prompt_length:].tolist(),
                       skip_special_tokens=True))[1] for row in input_ids]
            return torch.tensor(stopped, dtype=torch.bool, device=input_ids.device)

    return StoppingCriteriaList([FirstAnswerLine()])


def generation_fields(tokenizer, generated_ids):
    """Require the first answer boundary to occur only in the final token.

    HIDE excludes that unforwarded token, so scored states cannot include a later
    answer line. Text after the boundary within the last token is recorded.
    """
    ids = generated_ids.tolist() if hasattr(generated_ids, 'tolist') else generated_ids
    text = tokenizer.decode(ids, skip_special_tokens=True)
    answer, reached, suffix = first_answer_line(text)
    preceding = tokenizer.decode(ids[:-1], skip_special_tokens=True)
    if first_answer_line(preceding)[1]:
        raise ValueError('Answer boundary appeared before the final token; later-line states would be scored')
    return dict(generated_text=text, evaluated_text=answer,
                answer_boundary='first-line', answer_boundary_reached=reached,
                boundary_token_suffix=suffix, empty_answer=not bool(answer))
