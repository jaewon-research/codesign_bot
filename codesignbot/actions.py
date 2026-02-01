from enum import Enum


class CodesignActionType(Enum):
    """Action types for CodesignBot."""
    # System
    DO_NOTHING = "do_nothing"
    EXIT = "exit"
    
    # Friend requests
    SEND_FRIEND_REQUEST = "send_friend_request"
    ACCEPT_FRIEND_REQUEST = "accept_friend_request"
    REJECT_FRIEND_REQUEST = "reject_friend_request"
    
    # Questions
    ANSWER_QUESTION = "answer_question"
    SHARE_QUESTION = "share_question"
    
    # Notes
    CREATE_NOTE = "create_note"
    LIKE_NOTE = "like_note"
    COMMENT_ON_NOTE = "comment_on_note"

    # Comments (on posts)
    CREATE_COMMENT = "create_comment"
    LIKE_COMMENT = "like_comment"
    
    # Notifications
    GET_NOTIFICATIONS = "get_notifications"
    MARK_NOTIFICATION_READ = "mark_notification_read"

    @classmethod
    def get_default_actions(cls):
        """Default actions for codesign agents."""
        return [
            cls.DO_NOTHING,
            cls.SEND_FRIEND_REQUEST,
            cls.ACCEPT_FRIEND_REQUEST,
            cls.REJECT_FRIEND_REQUEST,
            cls.ANSWER_QUESTION,
            cls.SHARE_QUESTION,
            cls.CREATE_NOTE,
            cls.LIKE_NOTE,
            cls.COMMENT_ON_NOTE,
            cls.CREATE_COMMENT,
            cls.LIKE_COMMENT,
            cls.GET_NOTIFICATIONS,
            cls.MARK_NOTIFICATION_READ,
        ]