"""
Custom SocialAction class for CodesignBot.
Extends the base OASIS SocialAction with note-related actions.
"""
from typing import Any
from camel.toolkits import FunctionTool
from oasis.social_agent.agent_action import SocialAction
from oasis.social_platform.typing import ActionType


class CodesignSocialAction(SocialAction):
    """Extended SocialAction with note-related actions for CodesignBot."""

    def get_openai_function_list(self) -> list[FunctionTool]:
        """Return list of available action tools including custom note actions."""
        # Get base actions
        base_tools = super().get_openai_function_list()
        
        # Add custom note actions
        note_tools = [
            FunctionTool(self.create_note),
            FunctionTool(self.like_note),
            FunctionTool(self.comment_on_note),
            FunctionTool(self.send_friend_request),
            FunctionTool(self.accept_friend_request),
        ]
        
        return base_tools + note_tools

    async def create_note(self, content: str, visibility: str = "friends"):
        """Create a new note with the given content.

        Args:
            content (str): The content of the note to be created.
            visibility (str): Who can see this note - 'friends' or 'close_friends'.
                Default is 'friends'.

        Returns:
            dict: A dictionary indicating success and the note_id.
            Example: {'success': True, 'note_id': 50}
        """
        return await self.perform_action(
            (content, visibility), 
            "create_note"
        )

    async def like_note(self, note_id: int):
        """Like a note.

        Args:
            note_id (int): The ID of the note to like.

        Returns:
            dict: A dictionary indicating success.
            Example: {'success': True, 'like_id': 123}
        """
        return await self.perform_action(note_id, "like_note")

    async def comment_on_note(self, note_id: int, content: str):
        """Comment on a note.

        Args:
            note_id (int): The ID of the note to comment on.
            content (str): The comment content.

        Returns:
            dict: A dictionary indicating success and the comment_id.
            Example: {'success': True, 'comment_id': 456}
        """
        return await self.perform_action(
            (note_id, content),
            "comment_on_note"
        )

    async def send_friend_request(self, target_user_id: int, choice: str = "friend"):
        """Send a friend request to another user.

        Args:
            target_user_id (int): The user ID to send request to.
            choice (str): Type of friendship - 'friend' or 'close_friend'.
                Default is 'friend'.

        Returns:
            dict: A dictionary indicating success.
            Example: {'success': True, 'request_id': 789}
        """
        return await self.perform_action(
            (target_user_id, choice),
            "send_friend_request"
        )

    async def accept_friend_request(self, request_id: int, choice: str = "friend"):
        """Accept a pending friend request.

        Args:
            request_id (int): The ID of the friend request to accept.
            choice (str): Type of friendship to establish - 'friend' or 'close_friend'.
                Default is 'friend'.

        Returns:
            dict: A dictionary indicating success.
            Example: {'success': True}
        """
        return await self.perform_action(
            (request_id, choice),
            "accept_friend_request"
        )

