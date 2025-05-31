from django.urls import path
from .views import ChatBotView, ChatHistoryView
from .streaming_views import StreamingChatBotView

urlpatterns = [
    path('chat/', ChatBotView.as_view(), name='chat'),
    path('chat-stream/', StreamingChatBotView.as_view(), name='chat-stream'),
    path('chat-history/', ChatHistoryView.as_view(), name='chat-history'),
]
