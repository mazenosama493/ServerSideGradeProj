import logging
from datetime import datetime

logger = logging.getLogger('django.request')

class AccessLogMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        print("Middleware triggered!")  # Debug line
        response = self.get_response(request)
        
        remote_addr = request.META.get('REMOTE_ADDR', '-')
        print(f"Attempting to log: {remote_addr} {request.method} {request.path_info}")  # Debug line

        logger.info(
            '',
            extra={
                'remote_addr': remote_addr,
                'request_method': request.method,
                'path_info': request.path_info,
                'status_code': response.status_code,
            }
        )
        return response
        