from django.db import models
from django.contrib.auth.models import User

class ChatHistory(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    prompt = models.TextField()
    image = models.ImageField(upload_to='chat_images/', blank=True, null=True)
    response = models.TextField()
    source = models.CharField(max_length=20, default="unknown")  # e.g., 'desktop' or 'mobile'
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.source} - {self.prompt[:30]}..."
