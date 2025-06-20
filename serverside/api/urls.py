from django.urls import path
from .views import ChatBotView, ChatHistoryView, TranscribeAudioView

urlpatterns = [
    path('chat/', ChatBotView.as_view(), name='chat'),
    path('chat-history/', ChatHistoryView.as_view(), name='chat-history'),
    path('transcribe-audio/', TranscribeAudioView.as_view(), name='transcribe-audio'),
]
