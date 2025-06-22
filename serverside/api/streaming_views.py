# streaming_views.py
import json
import logging
import time
from django.http import StreamingHttpResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, JSONParser, FormParser
from .models import ChatHistory
import openai
from django.conf import settings
from dotenv import load_dotenv
import base64
from PIL import Image
import io
from rest_framework.permissions import IsAuthenticated
from rest_framework.decorators import permission_classes
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure API keys
openrouter_api_key = settings.OPENROUTER_API_KEY

DEFAULT_HTTP_HEADERS = {
    "HTTP-Referer": "https://eyeconic-chat.example",
    "X-Title": "Eyeconic Chat App",
}

class StreamingChatBotView(APIView):
    parser_classes = (MultiPartParser, JSONParser, FormParser)
    permission_classes = [IsAuthenticated]  # ✅ ADDED: Ensure user is authenticated to access this view

    def _get_relevant_history(self, user):  # ✅ ADDED: Accept user to filter chats
        """Get last 10 user-specific interactions to maintain context."""
        history = ChatHistory.objects.filter(user=user).order_by('-timestamp')[:10]  # ✅ MODIFIED: Only fetch user's chats
        context = []
        for chat in reversed(history):  # Reverse to get chronological order
            context.append(f"User: {chat.prompt}")
            context.append(f"Assistant: {chat.response}")
        return "\n".join(context)

    def prepare_image(self, image_data):
        """Convert image data to base64 for AI processing."""
        try:
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG", quality=85)
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
        except Exception as e:
            logger.error(f"Error preparing image: {str(e)}")
            raise ValueError(f"Error processing image: {str(e)}")

    def stream_response_generator(self, prompt, image_file=None, user=None):  # ✅ ADDED user param
        """Generator function that yields streaming response chunks."""
        try:
            # Set up OpenAI client for streaming
            session = openai.Client(
                api_key=openrouter_api_key,
                base_url="https://openrouter.ai/api/v1",
                default_headers=DEFAULT_HTTP_HEADERS
            )

            img_base64 = None

            # Handle image upload if present
            if image_file:
                try:
                    img_bytes = image_file.read()
                    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                    image_file.seek(0)  # Reset file pointer for later use
                except Exception as e:
                    logger.error(f"Error processing image: {str(e)}")
                    yield f"data: {json.dumps({'error': f'Error processing image: {str(e)}'})}\n\n"
                    return

            # ✅ MODIFIED: Pass user to history method
            chat_history = self._get_relevant_history(user)

            # System and user messages
            system_message = {
                "role": "system",
                "content": f"""You are Eyeconic, a professional AI assistant and advisor. Only introduce yourself as "I am Eyeconic, your AI assistant and advisor" when explicitly asked about your identity.

You have access to previous conversation history for context:
{chat_history}

Important instructions:
1. Maintain professionalism
2. Use chat history for context
3. Do not introduce yourself unless asked
4. Handle images when provided"""
            }

            if img_base64:
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
                user_message = {
                    "role": "user",
                    "content": prompt
                }

            yield f"data: {json.dumps({'type': 'connection', 'status': 'connected'})}\n\n"

            response_stream = session.chat.completions.create(
                model="opengvlab/internvl3-14b:free",
                messages=[system_message, user_message],
                stream=True,
                temperature=0.7,
                max_tokens=2048
            )

            complete_response = ""

            for chunk in response_stream:
                if chunk.choices[0].delta.content is not None:
                    content_chunk = chunk.choices[0].delta.content
                    complete_response += content_chunk
                    chunk_data = {
                        'type': 'content',
                        'content': content_chunk,
                        'complete': False
                    }
                    yield f"data: {json.dumps(chunk_data)}\n\n"

            yield f"data: {json.dumps({'type': 'complete', 'complete': True})}\n\n"

            # ✅ ADDED user to saved chat
            try:
                ChatHistory.objects.create(
                    user=user,  # ✅ associate with the current user
                    prompt=prompt,
                    image=image_file if image_file else None,
                    response=complete_response,
                    source="mobile"
                )
                logger.info(f"Saved streaming chat to history: {len(complete_response)} chars")
            except Exception as e:
                logger.error(f"Error saving to chat history: {str(e)}")
                yield f"data: {json.dumps({'type': 'error', 'error': 'Failed to save chat history'})}\n\n"

        except Exception as e:
            logger.error(f"Error in streaming response: {str(e)}")
            yield f"data: {json.dumps({'type': 'error', 'error': f'Server error: {str(e)}'})}\n\n"

    def post(self, request):
        """Handle streaming chat requests."""
        try:
            logger.info("Streaming chat request received")
            logger.info(f"Request data: {request.data}")

            prompt = request.data.get('prompt', '')
            if not prompt:
                return Response({"error": "No prompt provided"}, status=400)

            image_file = request.FILES.get('image', None)

            # ✅ Pass request.user to generator
            response = StreamingHttpResponse(
                self.stream_response_generator(prompt, image_file, request.user),  # ✅ user passed here
                content_type='text/event-stream'
            )

            # Headers for SSE
            response['Access-Control-Allow-Origin'] = '*'
            response['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
            response['Access-Control-Allow-Headers'] = 'Content-Type'
            response['Cache-Control'] = 'no-cache'
            response['X-Accel-Buffering'] = 'no'

            return response

        except Exception as e:
            logger.error(f"Error in StreamingChatBotView: {str(e)}")
            logger.exception("Full exception details:")
            return Response({"error": f"Server error: {str(e)}"}, status=500)

    def options(self, request):
        """Handle preflight CORS requests."""
        response = Response()
        response['Access-Control-Allow-Origin'] = '*'
        response['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response['Access-Control-Allow-Headers'] = 'Content-Type'
        return response
