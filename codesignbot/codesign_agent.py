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
        """Perform action decided by LLM based on environment observation."""
        # Get posts and environment info
        env_prompt = await self.env.to_text_prompt()
        user_msg = BaseMessage.make_user_message(
            role_name="User",
            content=(
                f"Please perform social media actions after observing the "
                f"platform environments. Notice that don't limit your "
                f"actions for example to just like the posts. "
                f"Here is your social media environment: {env_prompt}"))
        
        if self.interview_record:
            self.context_record = env_prompt
        
        response = await self.astep(user_msg)
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

