"""
Custom CrewAI callbacks to capture detailed agent execution in Arize.

This provides granular visibility into:
- Agent reasoning steps
- Tool calls within agents
- Task inputs and outputs
- LLM prompts and responses
- Agent decision-making process
"""
from typing import Any, Dict, List, Optional
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode


class ArizeCrewCallback:
    """
    Custom callback for CrewAI to capture detailed execution traces in Arize.

    This callback hooks into CrewAI's execution lifecycle to capture:
    - Task start/completion
    - Agent actions and thoughts
    - Tool invocations
    - LLM interactions
    """

    def __init__(self):
        self.tracer = trace.get_tracer("crewai-callbacks")
        self.active_spans = {}  # Track active spans by task/agent

    def on_task_start(self, task: Any, agent: Any) -> None:
        """Called when a task starts executing."""
        task_name = getattr(task, 'description', str(task))[:100]  # First 100 chars
        agent_role = getattr(agent, 'role', 'unknown_agent')

        span = self.tracer.start_span(
            f"task_execution_{agent_role}",
            attributes={
                "openinference.span.kind": "agent",
                "agent.role": agent_role,
                "agent.goal": getattr(agent, 'goal', 'unknown'),
                "task.description_preview": task_name,
                "task.id": str(id(task)),
                "input.value": task_name,
                "input.mime_type": "text/plain",
            }
        )

        # Store span for later completion
        self.active_spans[id(task)] = span

    def on_task_end(self, task: Any, output: Any) -> None:
        """Called when a task completes."""
        task_id = id(task)
        span = self.active_spans.get(task_id)

        if span:
            # Capture output
            output_str = str(output)[:5000]  # Limit to 5000 chars
            span.set_attribute("output.value", output_str)
            span.set_attribute("output.mime_type", "text/plain")
            span.set_attribute("task.output_length", len(output_str))
            span.set_status(Status(StatusCode.OK))
            span.end()

            # Clean up
            del self.active_spans[task_id]

    def on_agent_action(self, agent: Any, action: str, action_input: Any) -> None:
        """Called when an agent takes an action (e.g., uses a tool)."""
        agent_role = getattr(agent, 'role', 'unknown_agent')

        with self.tracer.start_as_current_span(
            f"agent_action_{action}",
            attributes={
                "openinference.span.kind": "tool",
                "agent.role": agent_role,
                "action.name": action,
                "action.input": str(action_input)[:1000],  # Limit input
                "input.value": str(action_input)[:1000],
                "input.mime_type": "text/plain",
            }
        ) as span:
            span.add_event("agent_action_started", {
                "action": action,
                "agent": agent_role
            })

    def on_agent_finish(self, agent: Any, output: Any) -> None:
        """Called when an agent finishes its work."""
        agent_role = getattr(agent, 'role', 'unknown_agent')

        with self.tracer.start_as_current_span(
            f"agent_finish_{agent_role}",
            attributes={
                "openinference.span.kind": "agent",
                "agent.role": agent_role,
                "output.value": str(output)[:2000],
                "output.mime_type": "text/plain",
            }
        ):
            pass

    def on_tool_start(self, tool_name: str, input_str: str) -> None:
        """Called when a tool is invoked by an agent."""
        with self.tracer.start_as_current_span(
            f"tool_{tool_name}",
            attributes={
                "openinference.span.kind": "tool",
                "tool.name": tool_name,
                "input.value": input_str[:1000],
                "input.mime_type": "text/plain",
            }
        ) as span:
            span.add_event("tool_invoked", {"tool_name": tool_name})

    def on_tool_end(self, tool_name: str, output: str) -> None:
        """Called when a tool completes."""
        # Note: In practice, you'd need to match this with the start span
        # This is a simplified version
        pass

    def on_llm_start(self, prompts: List[str], **kwargs) -> None:
        """Called when LLM is invoked."""
        with self.tracer.start_as_current_span(
            "llm_call",
            attributes={
                "openinference.span.kind": "llm",
                "llm.prompt_count": len(prompts),
                "input.value": prompts[0][:2000] if prompts else "",  # First prompt
                "input.mime_type": "text/plain",
            }
        ) as span:
            span.add_event("llm_invoked", {
                "prompt_count": len(prompts)
            })

    def on_llm_end(self, response: Any, **kwargs) -> None:
        """Called when LLM completes."""
        # Note: Would need to match with start span in real implementation
        pass

    def on_chain_start(self, chain_name: str, inputs: Dict[str, Any]) -> None:
        """Called when a chain starts (for LangChain chains used by CrewAI)."""
        with self.tracer.start_as_current_span(
            f"chain_{chain_name}",
            attributes={
                "openinference.span.kind": "chain",
                "chain.name": chain_name,
                "input.value": str(inputs)[:1000],
                "input.mime_type": "application/json",
            }
        ):
            pass

    def on_chain_end(self, chain_name: str, outputs: Dict[str, Any]) -> None:
        """Called when a chain completes."""
        # Note: Would need to match with start span
        pass

    def on_agent_thought(self, agent: Any, thought: str) -> None:
        """Called when an agent has a 'thought' (reasoning step)."""
        agent_role = getattr(agent, 'role', 'unknown_agent')

        # Add as an event to the current span
        current_span = trace.get_current_span()
        if current_span:
            current_span.add_event(
                "agent_thought",
                {
                    "agent.role": agent_role,
                    "thought": thought[:500],  # Limit thought length
                }
            )


class DetailedTaskCallback:
    """
    Alternative callback that focuses on capturing task-level details.
    Use this for simpler, task-focused instrumentation.
    """

    def __init__(self):
        self.tracer = trace.get_tracer("crewai-task-callbacks")

    def task_callback(self, task_output: Any) -> None:
        """
        Callback that's executed when a task completes.
        Can be passed directly to CrewAI Task objects.
        """
        with self.tracer.start_as_current_span(
            "task_callback_execution",
            attributes={
                "openinference.span.kind": "chain",
                "task.output": str(task_output)[:2000],
                "output.value": str(task_output)[:2000],
                "output.mime_type": "text/plain",
            }
        ) as span:
            span.add_event("task_completed", {
                "output_length": len(str(task_output))
            })
