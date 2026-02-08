"""
Custom SocialAgent for CodesignBot.
Extends the base OASIS SocialAgent with note-related actions.
"""
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Union

from camel.agents import ChatAgent
from camel.memories import MemoryRecord
from camel.messages import BaseMessage
from camel.models import BaseModelBackend, ModelManager
from camel.types import OpenAIBackendRole

from oasis.social_agent.agent_environment import SocialEnvironment
from oasis.social_platform import Channel
from oasis.social_platform.config import UserInfo
from oasis.social_platform.typing import ActionType

# Import our custom action class
from codesign_action import CodesignSocialAction
from codesign_platform import CodesignPlatform

if TYPE_CHECKING:
    from camel.toolkits import FunctionTool
    from oasis.social_agent.agent_graph import AgentGraph


class CodesignSocialAgent(ChatAgent):
    """Social agent with extended note-related actions for CodesignBot."""

    def __init__(self,
                 agent_id: int,
                 user_info: UserInfo,
                 channel: Channel = None,
                 user_info_template: str = None,
                 model: Optional[Union[BaseModelBackend,
                                       List[BaseModelBackend],
                                       ModelManager]] = None,
                 agent_graph: "AgentGraph" = None,
                 available_actions: list[ActionType] = None,
                 tools: Optional[List[Union["FunctionTool", Callable]]] = None,
                 max_iteration: int = 1,
                 interview_record: bool = False):
        self.social_agent_id = agent_id
        self.user_info = user_info
        self.channel = channel or Channel()
        
        # Use our custom CodesignSocialAction instead of base SocialAction
        self.env = SocialEnvironment(CodesignSocialAction(agent_id, self.channel))
        
        if user_info_template is None:
            system_message_content = self.user_info.to_system_message()
        else:
            system_message_content = self.user_info.to_custom_system_message(
                user_info_template)
        self.agent_graph = agent_graph
        system_message = BaseMessage.make_assistant_message(
            role_name="Social agent",
            content=system_message_content,
        )

        if not available_actions:
            self.action_tools = self.env.action.get_openai_function_list()
        else:
            all_tools = self.env.action.get_openai_function_list()
            all_possible_actions = [tool.func.__name__ for tool in all_tools]

            for action in available_actions:
                action_name = action.value if isinstance(
                    action, ActionType) else action
                if action_name not in all_possible_actions:
                    # Log but don't warn for our custom actions
                    pass
            self.action_tools = [
                tool for tool in all_tools if tool.func.__name__ in [
                    a.value if isinstance(a, ActionType) else a
                    for a in available_actions
                ]
            ]
        all_tools = (tools or []) + (self.action_tools or [])
        super().__init__(
            system_message=system_message,
            model=model,
            tools=all_tools,
        )
        self.max_iteration = max_iteration
        self.interview_record = interview_record

    async def perform_action_by_llm(self):
        """Perform action using two-step process: thinking then acting."""
        # Get posts and environment info
        env_prompt = await self.env.to_text_prompt()
        
        if self.interview_record:
            self.context_record = env_prompt
        
        # ==================== STEP 1: THINKING ====================
        # Agent reflects on the environment without taking action
        thinking_msg = BaseMessage.make_user_message(
            role_name="User",
            content=(
                f"You are browsing a social media platform. Take a moment to observe "
                f"and reflect on what you see.\n\n"
                f"Here is your social media environment:\n{env_prompt}\n\n"
                f"Based on your personality and interests, think about:\n"
                f"1. What posts catch your attention and why?\n"
                f"2. How do you feel about the content you're seeing?\n"
                f"3. What would feel most authentic for you to do right now?\n\n"
                f"Share your thoughts naturally, as if thinking to yourself. "
                f"Keep it brief (2-3 sentences)."
            ))
        
        # Call LLM for thinking - it should respond with text reflection
        # We don't need to disable tools; the prompt asks for reflection, not action
        try:
            thinking_response = await self.astep(thinking_msg)
            thoughts = thinking_response.msgs[0].content if thinking_response.msgs else ""
        except Exception as e:
            thoughts = f"(thinking interrupted: {e})"
        
        # Log the agent's thoughts
        agent_name = self.user_info.user_name if hasattr(self.user_info, 'user_name') else f"Agent {self.social_agent_id}"
        print(f"💭 [{agent_name}] thinking: {thoughts[:150]}{'...' if len(thoughts) > 150 else ''}", flush=True)
        
        # ==================== STEP 2: ACTING ====================
        # Agent takes action based on their thoughts
        action_msg = BaseMessage.make_user_message(
            role_name="User",
            content=(
                f"Based on your reflection:\n\"{thoughts}\"\n\n"
                f"Now take the social media action that feels most authentic to you. "
                f"You can create a post, like something, comment, send a friend request, "
                f"or do nothing if nothing feels right."
            ))
        
        # Call LLM with tools to take action
        response = await self.astep(action_msg)
        
        # ==================== STORE THOUGHT IN DATABASE ====================
        # Extract what action was taken from the response
        action_taken = None
        action_target_id = None
        action_result_id = None
        
        try:
            tool_calls = response.info.get('tool_calls', [])
            if tool_calls:
                first_call = tool_calls[0]
                action_taken = first_call.tool_name
                
                # Try to extract target ID from args (e.g., note_id for like/comment)
                args = first_call.args
                if isinstance(args, dict):
                    action_target_id = args.get('note_id') or args.get('post_id') or args.get('target_user_id')
                
                # Try to extract result ID from tool execution result
                # The result is in first_call.result or response.info.get('tool_call_results')
                result = getattr(first_call, 'result', None)
                if result is None:
                    # Try alternative location
                    tool_results = response.info.get('tool_call_results', [])
                    if tool_results:
                        result = tool_results[0]
                
                if isinstance(result, dict):
                    # For create_note: {"success": True, "note_id": X}
                    # For comment_on_note: {"success": True, "comment_id": X}
                    action_result_id = result.get('note_id') or result.get('comment_id')
                elif isinstance(result, str):
                    # Try to parse as JSON if it's a string
                    try:
                        import json
                        parsed = json.loads(result)
                        if isinstance(parsed, dict):
                            action_result_id = parsed.get('note_id') or parsed.get('comment_id')
                    except:
                        pass
        except Exception as e:
            pass  # If we can't extract action info, that's okay
        
        # Store the thought in the database
        CodesignPlatform.store_agent_thought(
            agent_id=self.social_agent_id,
            thought=thoughts,
            action_taken=action_taken,
            action_target_id=action_target_id,
            action_result_id=action_result_id
        )
        
        return response

    async def perform_action_by_hci(self) -> Any:
        """Perform action via human input (for controllable agents)."""
        env_prompt = await self.env.to_text_prompt()
        print(env_prompt)
        while True:
            act = input("Please input your action: ")
            args = input("Please input your arguments "
                         "(use ',' to separate different args): ")
            args_list = args.split(",")
            args_list = [arg.strip() for arg in args_list]

            # Make compatible with controllable=True settings
            tool_list = self.env.action.get_openai_function_list()
            tool_name_list = [tool.get_function_name() for tool in tool_list]
            if act in tool_name_list:
                for tool in tool_list:
                    if tool.get_function_name() == act:
                        result = tool.func(**dict(
                            zip(tool.get_parameter_names(), args_list)))
                        print(f"Action result: {result}")
                        return result
            else:
                print(f"Action {act} not found in available actions. "
                      f"Available: {tool_name_list}")

    def record_message(self, message: BaseMessage):
        """Record a message to memory."""
        memory_record = MemoryRecord(
            message=message, role_at_backend=OpenAIBackendRole.USER)
        self.memory.write_record(memory_record)

