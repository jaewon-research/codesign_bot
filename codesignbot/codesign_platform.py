import sqlite3
from oasis.social_platform import Channel
from oasis.social_platform.platform import Platform
from actions import CodesignActionType
from database import (
    send_friend_request, accept_friend_request, reject_friend_request, submit_question_response, 
    share_question_with_friend, create_note as create_note_db, like_note as like_note_db, get_note_owner, 
    create_comment as create_comment_db, like_comment as like_comment_db, get_user_notifications, get_comment_owner, 
    get_post_owner, mark_notification_read, create_notification, create_note_comment as create_note_comment_db
)

class CodesignPlatform(Platform):
    """Extended Platform with codesign-specific action handlers."""
    
    # Per-agent timestamps for realistic timing within timesteps
    # Set by simulation before each agent acts
    _agent_timestamps = {}
    
    @classmethod
    def set_agent_timestamp(cls, agent_id: int, timestamp: int):
        """Set the timestamp for a specific agent's actions."""
        cls._agent_timestamps[agent_id] = timestamp
    
    @classmethod
    def clear_agent_timestamps(cls):
        """Clear all agent timestamps (call at end of timestep)."""
        cls._agent_timestamps.clear()
    
    def get_time_for_agent(self, agent_id: int) -> int:
        """Get the timestamp to use for this agent's actions.
        
        Returns the agent's assigned timestamp if set, otherwise falls back to clock.
        """
        if agent_id in self._agent_timestamps:
            return self._agent_timestamps[agent_id]
        # Fallback to global clock
        return int(self.sandbox_clock.get_time_step())
    
    def _get_username(self, user_id):
        """Get username for logging purposes."""
        try:
            self.db_cursor.execute("SELECT user_name FROM user WHERE user_id = ?", (user_id,))
            result = self.db_cursor.fetchone()
            return result[0] if result else f"User#{user_id}"
        except:
            return f"User#{user_id}"
 
    # ==================== Main Simulation Loop ====================
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
            # Convert ActionType enum to string if needed
            action_name = action.value if hasattr(action, 'value') else str(action)
            action_function = getattr(self, action_name, None)
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

    # ==================== Refresh Recommended Posts ====================
    async def refresh(self, agent_id: int):
        """Override refresh to show notes instead of posts."""
        try:
            user_id = agent_id
            
            # Get notes visible to this agent based on connections
            # Include notes from friends and close friends
            note_query = """
                SELECT n.note_id, n.user_id, n.content, n.created_at, n.visibility,
                    (SELECT COUNT(*) FROM note_like WHERE note_id = n.note_id) as num_likes
                FROM note n
                WHERE n.user_id IN (
                    SELECT user2_id FROM connection WHERE user1_id = ?
                    UNION
                    SELECT user1_id FROM connection WHERE user2_id = ?
                )
                OR n.user_id = ?
                OR n.user_id = 0  -- Always include human user's notes
                ORDER BY n.created_at DESC
                LIMIT 10
            """
            self.db_cursor.execute(note_query, (user_id, user_id, user_id))
            notes = self.db_cursor.fetchall()
            
            results = []
            for note in notes:
                note_id, note_user_id, content, created_at, visibility, num_likes = note
                
                # Get comments for this note
                comments = []
                try:
                    self.db_cursor.execute("""
                        SELECT nc.comment_id, nc.user_id, nc.content, nc.created_at
                        FROM note_comment nc
                        WHERE nc.note_id = ?
                        ORDER BY nc.created_at ASC
                    """, (note_id,))
                    comments = [{"comment_id": c[0], "user_id": c[1], "content": c[2], "created_at": c[3]} 
                            for c in self.db_cursor.fetchall()]
                except:
                    pass
                
                results.append({
                    "post_id": note_id,  # Keep as post_id for compatibility with agent prompts
                    "user_id": note_user_id,
                    "content": content,
                    "created_at": created_at,
                    "num_likes": num_likes,
                    "num_dislikes": 0,
                    "num_shares": 0,
                    "num_reports": 0,
                    "comments": comments
                })
            
            return {"success": True, "posts": results}
        except Exception as e:
            return {"success": False, "error": str(e)}

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
            username = self._get_username(agent_id)
            target_name = self._get_username(target_user_id)
            print(f"🤝 [{username}] sent friend request to {target_name} (level: {friendship_level})", flush=True)
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
            username = self._get_username(agent_id)
            print(f"✅ [{username}] accepted friend request #{request_id} (level: {friendship_level})", flush=True)
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
            username = self._get_username(agent_id)
            print(f"❌ [{username}] rejected friend request #{request_id}", flush=True)
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
            username = self._get_username(agent_id)
            preview = response_text[:40] + "..." if len(response_text) > 40 else response_text
            print(f"❓ [{username}] answered question #{question_id}: \"{preview}\"", flush=True)
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
            username = self._get_username(agent_id)
            recipient_name = self._get_username(recipient_id)
            print(f"📤 [{username}] shared question #{question_id} with {recipient_name}", flush=True)
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
            # Get this agent's assigned timestamp (or fallback to clock)
            current_time = self.get_time_for_agent(agent_id)
            
            note_id = create_note_db(
                self.db_cursor,
                user_id=agent_id,
                content=content,
                visibility=visibility,
                created_at=current_time
            )
            self.db.commit()
            username = self._get_username(agent_id)
            preview = content[:60] + "..." if len(content) > 60 else content
            print(f"📝 [{username}] created note #{note_id} (t={current_time}): \"{preview}\"", flush=True)
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
            
            username = self._get_username(agent_id)
            owner_name = self._get_username(note_owner[0]) if note_owner else "unknown"
            print(f"❤️  [{username}] liked note #{note_id} by {owner_name}", flush=True)
            return {"success": True, "like_id": like_id}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def comment_on_note(self, agent_id, message):
        """Handle comment_on_note action.
        
        Args:
            agent_id: The user commenting
            message: (note_id, content)
        """
        note_id, content = message
        try:
            # Get this agent's assigned timestamp (or fallback to clock)
            current_time = self.get_time_for_agent(agent_id)
            
            comment_id = create_note_comment_db(
                self.db_cursor,
                user_id=agent_id,
                note_id=note_id,
                content=content,
                created_at=current_time
            )
            self.db.commit()
            
            # Notify note owner
            note_owner = get_note_owner(self.db_cursor, note_id)
            if note_owner and note_owner != agent_id:
                create_notification(
                    self.db_cursor,
                    recipient_id=note_owner,
                    sender_id=agent_id,
                    notification_type='note_comment',
                    content_text=content[:100],  # Truncate for notification
                    related_id=comment_id
                )
                self.db.commit()
            
            username = self._get_username(agent_id)
            preview = content[:50] + "..." if len(content) > 50 else content
            print(f"💬 [{username}] commented on note #{note_id} (t={current_time}): \"{preview}\"", flush=True)
            return {"success": True, "comment_id": comment_id}
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
            
            username = self._get_username(agent_id)
            preview = content[:50] + "..." if len(content) > 50 else content
            print(f"💬 [{username}] commented on post #{post_id}: \"{preview}\"", flush=True)
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
            
            username = self._get_username(agent_id)
            print(f"❤️  [{username}] liked comment #{comment_id}", flush=True)
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