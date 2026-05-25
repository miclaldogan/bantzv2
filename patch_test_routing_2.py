import sys

filepath = 'tests/core/test_routing_engine.py'
with open(filepath, 'r') as f:
    content = f.read()

old_code = """        with patch("bantz.core.routing_engine.registry") as mock_reg, \\
             patch("bantz.agent.planner.planner_agent") as mock_planner, \\
             patch("bantz.agent.executor.plan_executor") as mock_exec, \\
             patch("bantz.core.routing_engine.data_layer") as dl, \\
             patch("bantz.core.routing_engine.ollama") as mock_ollama, \\
             patch("bantz.core.routing_engine.finalize_plan") as mock_finalizer:
            mock_reg.names.return_value = ["shell", "weather"]
            mock_planner.decompose = AsyncMock(return_value=steps)
            mock_planner.format_itinerary.return_value = "Plan:\\n1. weather\\n2. shell"
            mock_exec.run = AsyncMock(return_value=exec_result)
            dl.conversations = MagicMock()
            mock_ollama.chat = AsyncMock(return_value="All done")
            mock_finalizer.return_value = "All done"

            from bantz.core.routing_engine import execute_plan
            result = _run(execute_plan("complex task", "complex task", {}))

        assert result is not None
        assert result.tool_used == "planner"
        assert "All done" in result.response"""

new_code = """        with patch("bantz.core.routing_engine.registry") as mock_reg, \\
             patch("bantz.agent.planner.planner_agent") as mock_planner, \\
             patch("bantz.agent.executor.plan_executor") as mock_exec, \\
             patch("bantz.core.routing_engine.data_layer") as dl, \\
             patch("bantz.core.routing_engine.ollama") as mock_ollama, \\
             patch("bantz.core.finalizer.finalize_plan") as mock_finalizer:
            mock_reg.names.return_value = ["shell", "weather"]
            mock_planner.decompose = AsyncMock(return_value=steps)
            mock_planner.format_itinerary.return_value = "Plan:\\n1. weather\\n2. shell"
            mock_exec.run = AsyncMock(return_value=exec_result)
            dl.conversations = MagicMock()
            mock_ollama.chat = AsyncMock(return_value="All done")
            mock_finalizer.return_value = "All done"

            from bantz.core.routing_engine import execute_plan
            result = _run(execute_plan("complex task", "complex task", {}))

        assert result is not None
        assert result.tool_used == "planner"
        assert "All done" in result.response"""

if old_code in content:
    content = content.replace(old_code, new_code)
    with open(filepath, 'w') as f:
        f.write(content)
    print("Patched part 2 successfully")
else:
    print("Could not find the target code to patch (part 2)")
