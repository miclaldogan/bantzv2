from bantz.interface.telegram_bot import _is_maintenance_spam
from tests.interface.test_telegram_llm import FakeBrainResult

result = FakeBrainResult(
    response="Workflow complete: 3/3 steps succeeded.\n✓ [a] ok\n✓ [b] ok\n✓ [c] ok",
    tool_used="maintenance",
)

print(_is_maintenance_spam(result))
