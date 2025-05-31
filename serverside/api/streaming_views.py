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

    def _get_relevant_history(self):
        """Get last 10 interactions to maintain context."""
        history = ChatHistory.objects.order_by('-timestamp')[:10]
        context = []
        for chat in reversed(history):  # Reverse to get chronological order
            context.append(f"User: {chat.prompt}")
            context.append(f"Assistant: {chat.response}")
        return "\n".join(context)

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

    def stream_response_generator(self, prompt, image_file=None):
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
                    # Convert image to base64 for API
                    img_bytes = image_file.read()
                    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                    image_file.seek(0)  # Reset file pointer for later use
                except Exception as e:
                    logger.error(f"Error processing image: {str(e)}")
                    yield f"data: {json.dumps({'error': f'Error processing image: {str(e)}'})}\n\n"
                    return

            # Get chat history for context
            chat_history = self._get_relevant_history()

            # Prepare messages for the API
            system_message = {
                "role": "system",
                "content": f"""You are Eyeconic, a professional AI assistant and advisor. Only introduce yourself as "I am Eyeconic, your AI assistant and advisor" when explicitly asked about your identity, name, or who you are. Otherwise, focus on directly answering questions and providing assistance without introducing yourself.

            You have access to previous conversation history for context:
            {chat_history}

            Important instructions:
            1. Maintain professionalism in all responses
            2. Remember and reference information users share about themselves from both current and previous conversations
            3. Use the chat history to maintain context and personalize responses
            4. Only introduce yourself when users specifically ask who you are
            5. Analyze and respond to questions about images when they are provided
            6. Acknowledge and build upon previous interactions when relevant"""
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

            # Send initial connection confirmation
            yield f"data: {json.dumps({'type': 'connection', 'status': 'connected'})}\n\n"

            # Make streaming API request
            response_stream = session.chat.completions.create(
                model="opengvlab/internvl3-14b:free",  # 14B model with image support
                messages=[system_message, user_message],
                stream=True,  # Enable streaming
                temperature=0.7,
                max_tokens=2048
            )

            # Collect the complete response for saving to history
            complete_response = ""

            # Stream the response chunks
            for chunk in response_stream:
                if chunk.choices[0].delta.content is not None:
                    content_chunk = chunk.choices[0].delta.content
                    complete_response += content_chunk

                    # Send each chunk to the client
                    chunk_data = {
                        'type': 'content',
                        'content': content_chunk,
                        'complete': False
                    }
                    yield f"data: {json.dumps(chunk_data)}\n\n"

            # Send completion signal
            yield f"data: {json.dumps({'type': 'complete', 'complete': True})}\n\n"

            # Save complete response to chat history
            try:
                ChatHistory.objects.create(
                    prompt=prompt,
                    image=image_file if image_file else None,
                    response=complete_response,
                    source="mobile"
                )
                logger.info(
                    f"Saved streaming chat to history: {len(complete_response)} characters")
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
                logger.error("No prompt provided")
                return Response({"error": "No prompt provided"}, status=400)

            logger.info(f"Processing prompt: {prompt}")
            image_file = None

            # Handle image upload if present
            if 'image' in request.FILES:
                image_file = request.FILES['image']
                # Create streaming response
                logger.info(f"Image file received: {image_file.name}")
            logger.info("Creating streaming response...")
            response = StreamingHttpResponse(
                self.stream_response_generator(prompt, image_file),
                content_type='text/event-stream'
            )

            # Set CORS and SSE headers (removed Connection header as it's hop-by-hop)
            response['Access-Control-Allow-Origin'] = '*'
            response['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
            response['Access-Control-Allow-Headers'] = 'Content-Type'
            response['Cache-Control'] = 'no-cache'
            # Disable nginx buffering for streaming
            response['X-Accel-Buffering'] = 'no'

            logger.info("Streaming response created successfully")
            return response

        except Exception as e:
            logger.error(f"Error in StreamingChatBotView: {str(e)}")
            logger.exception("Full exception details:")
            return Response(
                {"error": f"Server error: {str(e)}"},
                status=500
            )

    def options(self, request):
        """Handle preflight CORS requests."""
        response = Response()
        response['Access-Control-Allow-Origin'] = '*'
        response['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response['Access-Control-Allow-Headers'] = 'Content-Type'
        return response
