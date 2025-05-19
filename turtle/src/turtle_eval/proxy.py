# turtle_eval/proxy.py
class LCaseEvaluatorProxy:
    def __new__(cls, use_modified=True, *args, **kwargs):
        if use_modified:
            from turtle_eval.evaluator import EvaluatorAdapter  # Importación local aquí
            return EvaluatorAdapter(*args, **kwargs)