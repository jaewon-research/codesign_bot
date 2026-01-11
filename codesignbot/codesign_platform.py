import sqlite3
from oasis.social_platform import Channel
from oasis.social_platform.platform import Platform
from actions import CodesignActionType
from database import (
    send_friend_request, accept_friend_request, reject_friend_request, submit_question_response, 
    share_question_with_friend, create_note as create_note_db, like_note as like_note_db, get_note_owner, 
    create_comment as create_comment_db, like_comment as like_comment_db, get_user_notifications, get_comment_owner, 
    get_post_owner, mark_notification_read, create_notification
)

class CodesignPlatform(Platform):
    """Extended Platform with codesign-specific action handlers."""
 
    async def running(self):
        """Main loop - handles only CodesignActionType actions."""
        while True:
            message_id, data = await self.channel.receive_from()
            agent_id, message, action = data
            
            # Exit check
            if action == CodesignActionType.EXIT.value:
                if self.db_path == ":memory:":
                    dst = sqlite3.connect("codesign_backup.db")
                    with dst:
                        self.db.backup(dst)
                self.db_cursor.close()
                self.db.close()
                break
            
            # Find and call the handler method
            action_function = getattr(self, action, None)
            if action_function:
                try:
                    result = await self._call_action(action_function, agent_id, message)
                    await self.channel.send_to((message_id, agent_id, result))
                except Exception as e:
                    await self.channel.send_to((message_id, agent_id, {
                        "success": False, 
                        "error": str(e)
                    }))
            else:
                await self.channel.send_to((message_id, agent_id, {
                    "success": False,
                    "error": f"Unknown action: {action}"
                }))

    async def _call_action(self, action_function, agent_id, message):
        """Call the action function with the appropriate parameters."""
        func_code = action_function.__code__
        param_count = func_code.co_argcount

        if param_count == 1: # just self
            return await action_function()
        elif param_count == 2: # self, agent_id
            return await action_function(agent_id)
        else: # self, agent_id, message
            return await action_function(agent_id, message)

    # ==================== System Actions ====================
    
    async def do_nothing(self, agent_id):
        """Handle do_nothing action."""
        return {"success": True}
    
    # ==================== Friend Requests ====================

    async def send_friend_request(self, agent_id, message):
        """Handle send_friend_request action.
        
        Args:
            agent_id: The user sending the request
            message: (target_user_id, friendship_level)
        """
        target_user_id, friendship_level = message
        try:
            request_id = send_friend_request(
                self.db_cursor,
                requester_id=agent_id,
                requestee_id=target_user_id,
                requester_choice=friendship_level
            )
            self.db.commit()
            return {"success": True, "request_id": request_id}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def accept_friend_request(self, agent_id, message):
        """Handle accept_friend_request action.
        
        Args:
            agent_id: The user accepting the request
            message: (request_id, friendship_level)
        """
        request_id, friendship_level = message
        try:
            result = accept_friend_request(
                self.db_cursor,
                request_id=request_id,
                requestee_choice=friendship_level
            )
            self.db.commit()
            return {"success": result}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def reject_friend_request(self, agent_id, message):
        """Handle reject_friend_request action.
        
        Args:
            agent_id: The user rejecting the request
            message: request_id
        """
        request_id = message
        try:
            result = reject_friend_request(self.db_cursor, request_id)
            self.db.commit()
            return {"success": result}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # ==================== Question Actions ====================
    
    async def answer_question(self, agent_id, message):
        """Handle answer_question action.
        
        Args:
            agent_id: The user answering
            message: (question_id, response_text)
        """
        question_id, response_text = message
        try:
            response_id = submit_question_response(
                self.db_cursor,
                self.db,
                user_id=agent_id,
                daily_question_id=question_id,
                response_text=response_text
            )
            return {"success": True, "response_id": response_id}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def share_question(self, agent_id, message):
        """Handle share_question action.
        
        Args:
            agent_id: The user sharing
            message: (question_id, recipient_id)
        """
        question_id, recipient_id = message
        try:
            share_id, notification_id = share_question_with_friend(
                self.db_cursor,
                sender_id=agent_id,
                recipient_id=recipient_id,
                daily_question_id=question_id
            )
            self.db.commit()
            return {
                "success": True, 
                "share_id": share_id,
                "notification_id": notification_id
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # ==================== Note Actions ====================
    
    async def create_note(self, agent_id, message):
        """Handle create_note action.
        
        Args:
            agent_id: The user creating the note
            message: (content, visibility)
        """
        content, visibility = message
        try:
            note_id = create_note_db(
                self.db_cursor,
                user_id=agent_id,
                content=content,
                visibility=visibility
            )
            self.db.commit()
            return {"success": True, "note_id": note_id}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def like_note(self, agent_id, message):
        """Handle like_note action.
        
        Args:
            agent_id: The user liking
            message: note_id
        """
        note_id = message
        try:
            like_id = like_note_db(
                self.db_cursor,
                user_id=agent_id,
                note_id=note_id
            )
            self.db.commit()
            
            # Notify note owner
            note_owner = get_note_owner(self.db_cursor, note_id)
            if note_owner and note_owner[0] != agent_id:
                create_notification(
                    self.db_cursor,
                    recipient_id=note_owner[0],
                    sender_id=agent_id,
                    notification_type='like_note',
                    content_text='liked your note',
                    related_id=note_id
                )
                self.db.commit()
            
            return {"success": True, "like_id": like_id}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # ==================== Comment Actions ====================
    
    async def create_comment(self, agent_id, message):
        """Handle create_comment action.
        
        Args:
            agent_id: The user commenting
            message: (post_id, content)
        """
        post_id, content = message
        try:
            comment_id = create_comment_db(
                self.db_cursor,
                user_id=agent_id,
                post_id=post_id,
                content=content
            )
            self.db.commit()
            
            # Notify post owner
            post_owner = get_post_owner(self.db_cursor, post_id)
            if post_owner and post_owner[0] != agent_id:
                create_notification(
                    self.db_cursor,
                    recipient_id=post_owner[0],
                    sender_id=agent_id,
                    notification_type='comment',
                    content_text=content[:100],  # Truncate for notification
                    related_id=comment_id
                )
                self.db.commit()
            
            return {"success": True, "comment_id": comment_id}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def like_comment(self, agent_id, message):
        """Handle like_comment action.
        
        Args:
            agent_id: The user liking
            message: comment_id
        """
        comment_id = message
        try:
            like_id = like_comment_db(
                self.db_cursor,
                user_id=agent_id,
                comment_id=comment_id
            )
            self.db.commit()
            
            # Notify comment owner
            comment_owner = get_comment_owner(self.db_cursor, comment_id)
            if comment_owner and comment_owner[0] != agent_id:
                create_notification(
                    self.db_cursor,
                    recipient_id=comment_owner[0],
                    sender_id=agent_id,
                    notification_type='like_comment',
                    content_text='liked your comment',
                    related_id=comment_id
                )
                self.db.commit()
            
            return {"success": True, "like_id": like_id}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # ==================== Notification Actions ====================
    
    async def get_notifications(self, agent_id):
        """Handle get_notifications action.
        
        Args:
            agent_id: The user getting notifications
        """
        try:
            notifications = get_user_notifications(
                self.db_cursor, 
                user_id=agent_id,
                unread_only=True
            )
            return {"success": True, "notifications": notifications}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def mark_notification_read(self, agent_id, message):
        """Handle mark_notification_read action.
        
        Args:
            agent_id: The user
            message: notification_id
        """
        notification_id = message
        try:
            result = mark_notification_read(self.db_cursor, notification_id)
            self.db.commit()
            return {"success": result}
        except Exception as e:
            return {"success": False, "error": str(e)}