from oasis.social_platform.platform import Platform
from .database import (
    send_friend_request, accept_friend_request, reject_friend_request, submit_question_response, 
    share_question_with_friend, create_note, like_note, create_comment, like_comment, get_user_notifications, 
    mark_notification_read, create_notification
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
                    func_code = action_function.__code__
                    param_count = func_code.co_argcount
                    
                    if param_count == 1:  # just self
                        result = await action_function()
                    elif param_count == 2:  # self, agent_id
                        result = await action_function(agent_id)
                    else:  # self, agent_id, message
                        result = await action_function(agent_id, message)
                    
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
            # TODO: Add create_note to database.py
            self.db_cursor.execute("""
                INSERT INTO note (user_id, content, visibility)
                VALUES (?, ?, ?)
            """, (agent_id, content, visibility))
            note_id = self.db_cursor.lastrowid
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
            # TODO: Add like_note to database.py
            self.db_cursor.execute("""
                INSERT INTO note_like (user_id, note_id)
                VALUES (?, ?)
            """, (agent_id, note_id))
            like_id = self.db_cursor.lastrowid
            self.db.commit()
            
            # Notify note owner
            self.db_cursor.execute(
                "SELECT user_id FROM note WHERE note_id = ?", (note_id,)
            )
            note_owner = self.db_cursor.fetchone()
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
            self.db_cursor.execute("""
                INSERT INTO comment (user_id, post_id, content, created_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            """, (agent_id, post_id, content))
            comment_id = self.db_cursor.lastrowid
            self.db.commit()
            
            # Notify post owner
            self.db_cursor.execute(
                "SELECT user_id FROM post WHERE post_id = ?", (post_id,)
            )
            post_owner = self.db_cursor.fetchone()
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
            self.db_cursor.execute("""
                INSERT INTO comment_like (user_id, comment_id)
                VALUES (?, ?)
            """, (agent_id, comment_id))
            like_id = self.db_cursor.lastrowid
            self.db.commit()
            
            # Notify comment owner
            self.db_cursor.execute(
                "SELECT user_id FROM comment WHERE comment_id = ?", (comment_id,)
            )
            comment_owner = self.db_cursor.fetchone()
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