"""quant-regress — task-level regression testing for quantized models.

Perplexity and token-level metrics do not reliably detect the damage
quantization does at the margins. This package measures task accuracy
before and after quantization so a regression fails a build instead of
shipping.
"""

__version__ = "1.0.3"

from .harness import QuantHarness, ComparisonResult, Verdict

__all__ = ["QuantHarness", "ComparisonResult", "Verdict", "__version__"]
