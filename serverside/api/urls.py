from django.urls import path
from .views import ChatBotView, ChatHistoryView,DeleteChatView,TranscribeAudioView
from .streaming_views import StreamingChatBotView

urlpatterns = [
    path('chat/', ChatBotView.as_view(), name='chat'),
    path('chat-stream/', StreamingChatBotView.as_view(), name='chat-stream'),
    path('chat-history/', ChatHistoryView.as_view(), name='chat-history'),
    path('chat/<int:chat_id>/delete/', DeleteChatView.as_view(), name='delete-chat'),
    path('transcribe-audio/', TranscribeAudioView.as_view(), name='transcribe-audio'),
]
