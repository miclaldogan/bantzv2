import sys

filepath = 'tests/agent/test_planner.py'
with open(filepath, 'r') as f:
    content = f.read()

old_code = """        with patch("bantz.agent.planner.planner_agent") as mock_planner, \\
             patch("bantz.agent.executor.plan_executor") as mock_executor, \\
             patch("bantz.core.brain.data_layer") as mock_dal, \\
             patch("bantz.core.routing_engine.data_layer") as dal_re, \\
             patch("bantz.core.brain.ollama") as mock_ollama, \\
             patch("bantz.core.routing_engine.ollama") as ollama_re, \\
             patch("bantz.core.brain.cot_route") as mock_cot, \\
             patch("bantz.core.routing_engine.finalize_plan") as mock_finalizer:"""

new_code = """        with patch("bantz.agent.planner.planner_agent") as mock_planner, \\
             patch("bantz.agent.executor.plan_executor") as mock_executor, \\
             patch("bantz.core.brain.data_layer") as mock_dal, \\
             patch("bantz.core.routing_engine.data_layer") as dal_re, \\
             patch("bantz.core.brain.ollama") as mock_ollama, \\
             patch("bantz.core.routing_engine.ollama") as ollama_re, \\
             patch("bantz.core.brain.cot_route") as mock_cot, \\
             patch("bantz.core.finalizer.finalize_plan") as mock_finalizer:"""

content = content.replace(old_code, new_code)

old_code_3 = """    @pytest.mark.asyncio
    async def test_simple_request_bypasses_planner(self):
        \"\"\"Simple single-tool request should NOT trigger planner (cot_route returns 'tool').\"\"\"
        with patch("bantz.core.brain.cot_route") as mock_cot, \\
             patch("bantz.core.brain.data_layer") as mock_dal, \\
             patch("bantz.core.routing_engine.data_layer") as dal_re, \\
             patch("bantz.core.brain.ollama") as mock_ollama, \\
             patch("bantz.core.routing_engine.ollama") as ollama_re, \\
             patch("bantz.core.routing_engine.finalize_plan") as mock_finalizer:"""

new_code_3 = """    @pytest.mark.asyncio
    async def test_simple_request_bypasses_planner(self):
        \"\"\"Simple single-tool request should NOT trigger planner (cot_route returns 'tool').\"\"\"
        with patch("bantz.core.brain.cot_route") as mock_cot, \\
             patch("bantz.core.brain.data_layer") as mock_dal, \\
             patch("bantz.core.routing_engine.data_layer") as dal_re, \\
             patch("bantz.core.brain.ollama") as mock_ollama, \\
             patch("bantz.core.routing_engine.ollama") as ollama_re, \\
             patch("bantz.core.finalizer.finalize_plan") as mock_finalizer:"""

content = content.replace(old_code_3, new_code_3)

old_code_4 = """        with patch("bantz.agent.planner.planner_agent") as mock_planner, \\
             patch("bantz.agent.executor.plan_executor") as mock_executor, \\
             patch("bantz.core.brain.data_layer") as mock_dal, \\
             patch("bantz.core.routing_engine.data_layer") as dal_re, \\
             patch("bantz.core.brain.ollama") as mock_ollama, \\
             patch("bantz.core.routing_engine.ollama") as ollama_re, \\
             patch("bantz.core.workflow.workflow_engine") as mock_wf, \\
             patch("bantz.core.brain.cot_route") as mock_cot, \\
             patch("bantz.core.routing_engine.finalize_plan") as mock_finalizer:"""

new_code_4 = """        with patch("bantz.agent.planner.planner_agent") as mock_planner, \\
             patch("bantz.agent.executor.plan_executor") as mock_executor, \\
             patch("bantz.core.brain.data_layer") as mock_dal, \\
             patch("bantz.core.routing_engine.data_layer") as dal_re, \\
             patch("bantz.core.brain.ollama") as mock_ollama, \\
             patch("bantz.core.routing_engine.ollama") as ollama_re, \\
             patch("bantz.core.workflow.workflow_engine") as mock_wf, \\
             patch("bantz.core.brain.cot_route") as mock_cot, \\
             patch("bantz.core.finalizer.finalize_plan") as mock_finalizer:"""

content = content.replace(old_code_4, new_code_4)

with open(filepath, 'w') as f:
    f.write(content)
print("Patched planner test successfully")
