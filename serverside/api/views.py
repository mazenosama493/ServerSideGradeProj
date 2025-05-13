# views.py

import google.generativeai as genai
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, JSONParser
from .models import ChatHistory
from .serializers import ChatHistorySerializer
from PIL import Image
import io
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
# Configure Gemini API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

class ChatBotView(APIView):
    parser_classes = [MultiPartParser, JSONParser]

    def post(self, request):
        prompt = request.data.get("prompt")
        image_file = request.FILES.get("image")

        if not prompt:
            return Response({"error": "Prompt is required"}, status=400)

        try:
            if image_file:
                image_bytes = image_file.read()
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

                model = genai.GenerativeModel(model_name="gemini-1.5-flash")
                response = model.generate_content([prompt, image])
            else:
                model = genai.GenerativeModel(model_name="gemini-1.5-flash")
                response = model.generate_content(prompt)

            ChatHistory.objects.create(
                prompt=prompt,
                image=image_file if image_file else None,
                response=response.text,
                source="desktop" 
            )

            return Response({"response": response.text})
        except Exception as e:
            return Response({"error": str(e)}, status=500)
        
    
class ChatHistoryView(APIView):
    def get(self,request):
        chats = ChatHistory.objects.all().order_by('-timestamp') 
        serializer = ChatHistorySerializer(chats, many=True)
        return Response(serializer.data)

