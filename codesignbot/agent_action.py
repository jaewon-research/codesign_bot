from typing import Any
from camel.toolkits import FunctionTool
from oasis.social_platform.channel import Channel

from .actions import CodesignActionType


class CodesignAction:
    """Standalone actions for CodesignBot (no base Twitter/Reddit actions)."""
    
    def __init__(self, agent_id: int, channel: Channel):
        self.agent_id = agent_id
        self.channel = channel
    
    def get_openai_function_list(self) -> list[FunctionTool]:
        """Return only codesign-specific action functions."""
        return [
            FunctionTool(func) for func in [
                self.send_friend_request,
                self.accept_friend_request,
                self.reject_friend_request,
                self.answer_question,
                self.share_question,
                self.create_note,
                self.like_note,
                self.like_comment,
                self.get_notifications,
                self.mark_notification_read,
                self.do_nothing,
            ]
        ]
    
    async def perform_action(self, message: Any, action_type: str):
        """Send action to platform and wait for response."""
        message_id = await self.channel.write_to_receive_queue(
            (self.agent_id, message, action_type))
        response = await self.channel.read_from_send_queue(message_id)
        return response[2]
    
    # ==================== Basic ====================
    
    async def do_nothing(self):
        """Perform no action this turn."""
        return await self.perform_action(None, CodesignActionType.DO_NOTHING.value)
    
    # ==================== Friend Requests ====================
    
    async def send_friend_request(self, user_id: int, friendship_level: str = "friend"):
        """Send a friend request to another user.
        
        Args:
            user_id: The ID of the user to send a friend request to.
            friendship_level: Either 'friend' or 'close_friend'.
        """
        message = (user_id, friendship_level)
        return await self.perform_action(message, CodesignActionType.SEND_FRIEND_REQUEST.value)
    
    async def accept_friend_request(self, request_id: int, friendship_level: str = "friend"):
        """Accept a pending friend request.
        
        Args:
            request_id: The ID of the friend request to accept.
            friendship_level: Either 'friend' or 'close_friend'.
        """
        message = (request_id, friendship_level)
        return await self.perform_action(message, CodesignActionType.ACCEPT_FRIEND_REQUEST.value)
    
    async def reject_friend_request(self, request_id: int):
        """Reject a pending friend request."""
        return await self.perform_action(request_id, CodesignActionType.REJECT_FRIEND_REQUEST.value)
    
    # ==================== Questions ====================
    
    async def answer_question(self, question_id: int, response_text: str):
        """Answer a daily question.
        
        Args:
            question_id: The ID of the question to answer.
            response_text: Your response to the question.
        """
        message = (question_id, response_text)
        return await self.perform_action(message, CodesignActionType.ANSWER_QUESTION.value)
    
    async def share_question(self, question_id: int, recipient_id: int):
        """Share a question with another user.
        
        Args:
            question_id: The ID of the question to share.
            recipient_id: The user to share it with.
        """
        message = (question_id, recipient_id)
        return await self.perform_action(message, CodesignActionType.SHARE_QUESTION.value)
    
    # ==================== Notes ====================
    
    async def create_note(self, content: str, visibility: str = "friends"):
        """Create a note (like a post but for friends).
        
        Args:
            content: The note content.
            visibility: 'friends', 'close_friends', or 'public'.
        """
        message = (content, visibility)
        return await self.perform_action(message, CodesignActionType.CREATE_NOTE.value)
    
    async def like_note(self, note_id: int):
        """Like a friend's note."""
        return await self.perform_action(note_id, CodesignActionType.LIKE_NOTE.value)

    # ==================== Comments ====================

    async def create_comment(self, post_id: int, content: str):
        """Create a new comment for a post."""
        message = (post_id, content)
        return await self.perform_action(message, CodesignActionType.CREATE_COMMENT.value)

    async def like_comment(self, comment_id: int):
        """Like a comment."""
        return await self.perform_action(comment_id, CodesignActionType.LIKE_COMMENT.value)
    
    # ==================== Notifications ====================
    
    async def get_notifications(self):
        """Get all unread notifications."""
        return await self.perform_action(None, CodesignActionType.GET_NOTIFICATIONS.value)