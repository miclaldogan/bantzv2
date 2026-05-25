import sys

filepath = 'tests/core/test_routing_engine.py'
with open(filepath, 'r') as f:
    content = f.read()

old_code = """    def test_execute_plan_without_history_still_works(self):
        \"\"\"execute_plan without recent_history defaults to None (#212 backward compat).\"\"\"
        with patch("bantz.core.routing_engine.registry") as mock_reg, \\
             patch("bantz.agent.planner.planner_agent") as mock_planner, \\
             patch("bantz.core.routing_engine.data_layer") as dl:
            mock_reg.names.return_value = ["shell"]
            mock_planner.decompose = AsyncMock(return_value=[])
            dl.conversations = MagicMock()

            from bantz.core.routing_engine import execute_plan
            result = _run(execute_plan("hello", "hello", {}))

        assert result is None
        call_kwargs = mock_planner.decompose.call_args
        assert call_kwargs.kwargs.get("recent_history") is None"""

new_code = """    def test_execute_plan_without_history_still_works(self):
        \"\"\"execute_plan without recent_history defaults to None (#212 backward compat).\"\"\"
        with patch("bantz.core.routing_engine.registry") as mock_reg, \\
             patch("bantz.agent.planner.planner_agent") as mock_planner, \\
             patch("bantz.core.routing_engine.data_layer") as dl:
            mock_reg.names.return_value = ["shell"]
            mock_planner.decompose = AsyncMock(return_value=[])
            dl.conversations = MagicMock()

            from bantz.core.routing_engine import execute_plan
            result = _run(execute_plan("hello", "hello", {}))

        assert result is not None
        assert "couldn't break that into actionable steps" in result.response
        call_kwargs = mock_planner.decompose.call_args
        assert call_kwargs.kwargs.get("recent_history") is None"""

if old_code in content:
    content = content.replace(old_code, new_code)
    with open(filepath, 'w') as f:
        f.write(content)
    print("Patched part 3 successfully")
else:
    print("Could not find the target code to patch (part 3)")
