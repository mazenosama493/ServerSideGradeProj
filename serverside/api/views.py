# views.py
import google.generativeai as genai
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, JSONParser, FormParser
from .models import ChatHistory
from .serializers import ChatHistorySerializer
from PIL import Image
import io
import json
import os
import uuid
import base64
import openai
import logging
from django.conf import settings
from dotenv import load_dotenv
from django.core.files import File
import tempfile

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure API keys
gemini_api_key = settings.GEMINI_API_KEY
openrouter_api_key = settings.OPENROUTER_API_KEY

# Configure OpenRouter client
openai.api_key = openrouter_api_key
openai.base_url = "https://openrouter.ai/api/v1"

DEFAULT_HTTP_HEADERS = {
    "HTTP-Referer": "https://eyeconic-chat.example",
    "X-Title": "Eyeconic Chat App",
}


class ChatBotView(APIView):
    parser_classes = [MultiPartParser, JSONParser, FormParser]

    def __del__(self):
        """Clean up any temporary files when the view is destroyed."""
        try:
            temp_dir = tempfile.gettempdir()
            for filename in os.listdir(temp_dir):
                if filename.endswith('.jpeg') and os.path.isfile(os.path.join(temp_dir, filename)):
                    try:
                        os.remove(os.path.join(temp_dir, filename))
                    except:
                        pass  # Ignore errors on cleanup
        except:
            pass  # We don't want cleanup to cause issues

    def prepare_image(self, image_data):
        """Convert image data to base64 for AI processing."""
        try:
            # Convert to RGB format
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG", quality=85)
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
        except Exception as e:
            logger.error(f"Error preparing image: {str(e)}")
            raise ValueError(f"Error processing image: {str(e)}")

    def post(self, request):
        try:
            # Get the prompt from the request
            prompt = request.data.get("prompt", "")
            if not prompt:
                return Response({"error": "Prompt is required"}, status=400)

            # Process image if it exists
            image_file = request.FILES.get("image")
            img_base64 = None

            if image_file:
                try:
                    # Read image data
                    image_data = image_file.read()

                    # Create a temporary file with a proper name
                    image_name = f"{uuid.uuid4()}.jpeg"
                    temp_file = tempfile.NamedTemporaryFile(
                        delete=False, suffix='.jpeg')
                    temp_file.write(image_data)
                    temp_file.flush()
                    temp_file.close()

                    # Create a File object that is compatible with ImageField
                    image_file = File(open(temp_file.name, 'rb'))
                    image_file.name = image_name  # Set the name explicitly

                    # Prepare image for AI
                    img_base64 = self.prepare_image(image_data)
                except ValueError as e:
                    return Response({"error": str(e)}, status=400)

            # Set up OpenAI client
            session = openai.Client(
                api_key=openrouter_api_key,
                base_url="https://openrouter.ai/api/v1",
                default_headers=DEFAULT_HTTP_HEADERS
            )

            # Prepare messages for the API
            system_message = {
                "role": "system",
                "content": "You are Eyeconic, an AI assistant and advisor. Always introduce yourself as \"I am Eyeconic, your AI assistant and advisor\" when asked about your identity. You can analyze images and respond to questions about them."
            }

            if img_base64:
                # Image + text request
                user_message = {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_base64}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            else:
                # Text-only request
                user_message = {
                    "role": "user",
                    "content": prompt
                }

            # Make API request
            response = session.chat.completions.create(
                model="qwen/qwen2.5-vl-3b-instruct:free",
                messages=[system_message, user_message]
            )

            result_text = response.choices[0].message.content

            # Save to chat history
            ChatHistory.objects.create(
                prompt=prompt,
                image=image_file if image_file else None,
                response=result_text,
                source="mobile"  # Since we're focusing on mobile-first approach
            )

            return Response({"response": result_text})

        except Exception as e:
            logger.error(f"Error in ChatBotView: {str(e)}")
            return Response(
                {"error": f"Server error: {str(e)}"},
                status=500
            )


class ChatHistoryView(APIView):
    def get(self, request):
        try:
            chats = ChatHistory.objects.all().order_by("-timestamp")
            serializer = ChatHistorySerializer(chats, many=True)
            return Response(serializer.data)
        except Exception as e:
            logger.error(f"Error in ChatHistoryView: {str(e)}")
            return Response(
                {"error": f"Server error: {str(e)}"},
                status=500
            )
