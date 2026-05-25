import sys

filepath = 'tests/core/test_routing_engine.py'
with open(filepath, 'r') as f:
    content = f.read()

old_code = """    def test_returns_none_for_simple_request(self):
        with patch("bantz.core.routing_engine.registry") as mock_reg, \\
             patch("bantz.agent.planner.planner_agent") as mock_planner, \\
             patch("bantz.core.routing_engine.data_layer") as dl:
            mock_reg.names.return_value = ["shell", "weather"]
            mock_planner.decompose = AsyncMock(return_value=[])  # no steps
            dl.conversations = MagicMock()
            from bantz.core.routing_engine import execute_plan
            result = _run(execute_plan("hello", "hello", {}))
        assert result is not None
        assert "couldn't break that into actionable steps" in result.response"""

new_code = """    def test_returns_none_for_simple_request(self):
        with patch("bantz.core.routing_engine.registry") as mock_reg, \\
             patch("bantz.agent.planner.planner_agent") as mock_planner, \\
             patch("bantz.core.routing_engine.data_layer") as mock_dl:
            mock_reg.names.return_value = ["shell", "weather"]
            mock_planner.decompose = AsyncMock(return_value=[])  # no steps
            mock_dl.conversations = MagicMock()

            from bantz.core.routing_engine import execute_plan
            result = _run(execute_plan("hello", "hello", {}))

        assert result is not None
        assert "couldn't break that into actionable steps" in result.response"""

if old_code in content:
    content = content.replace(old_code, new_code)

    old_code_2 = """        with patch("bantz.core.routing_engine.registry") as mock_reg, \\
             patch("bantz.agent.planner.planner_agent") as mock_planner, \\
             patch("bantz.agent.executor.plan_executor") as mock_exec, \\
             patch("bantz.core.routing_engine.data_layer") as dl, \\
             patch("bantz.core.routing_engine.ollama") as mock_ollama, \\
             patch("bantz.core.finalizer.finalize_plan") as mock_finalizer:
            mock_reg.names.return_value = ["shell", "weather"]
            mock_planner.decompose = AsyncMock(return_value=steps)
            mock_planner.format_itinerary.return_value = "Plan:\\n1. weather\\n2. shell"
            mock_exec.run = AsyncMock(return_value=exec_result)
            dl.conversations = MagicMock()"""

    new_code_2 = """        with patch("bantz.core.routing_engine.registry") as mock_reg, \\
             patch("bantz.agent.planner.planner_agent") as mock_planner, \\
             patch("bantz.agent.executor.plan_executor") as mock_exec, \\
             patch("bantz.core.routing_engine.data_layer") as mock_dl, \\
             patch("bantz.core.routing_engine.ollama") as mock_ollama, \\
             patch("bantz.core.finalizer.finalize_plan") as mock_finalizer:
            mock_reg.names.return_value = ["shell", "weather"]
            mock_planner.decompose = AsyncMock(return_value=steps)
            mock_planner.format_itinerary.return_value = "Plan:\\n1. weather\\n2. shell"
            mock_exec.run = AsyncMock(return_value=exec_result)
            mock_dl.conversations = MagicMock()"""

    content = content.replace(old_code_2, new_code_2)

    old_code_3 = """    def test_execute_plan_without_history_still_works(self):
        \"\"\"execute_plan without recent_history defaults to None (#212 backward compat).\"\"\"
        with patch("bantz.core.routing_engine.registry") as mock_reg, \\
             patch("bantz.agent.planner.planner_agent") as mock_planner, \\
             patch("bantz.core.routing_engine.data_layer") as dl:
            mock_reg.names.return_value = ["shell"]
            mock_planner.decompose = AsyncMock(return_value=[])
            dl.conversations = MagicMock()"""

    new_code_3 = """    def test_execute_plan_without_history_still_works(self):
        \"\"\"execute_plan without recent_history defaults to None (#212 backward compat).\"\"\"
        with patch("bantz.core.routing_engine.registry") as mock_reg, \\
             patch("bantz.agent.planner.planner_agent") as mock_planner, \\
             patch("bantz.core.routing_engine.data_layer") as mock_dl:
            mock_reg.names.return_value = ["shell"]
            mock_planner.decompose = AsyncMock(return_value=[])
            mock_dl.conversations = MagicMock()"""

    content = content.replace(old_code_3, new_code_3)

    with open(filepath, 'w') as f:
        f.write(content)
    print("Patched part 4 successfully")
else:
    print("Could not find the target code to patch (part 4)")
