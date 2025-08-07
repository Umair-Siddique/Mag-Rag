
from flask import  request,current_app

def _extract_access_token() -> str | None:
    """
    Pull the JWT from the Authorization header:
        Authorization: Bearer <token>
    Return None if the header is missing or malformed.
    """
    auth_header = request.headers.get('Authorization', '')
    if auth_header.startswith('Bearer '):
        return auth_header.split(' ', 1)[1].strip() or None
    return None

def authenticate_user() -> object | None:
    """
    Authenticate the incoming request by validating the JWT against Supabase.
    Returns the Supabase user object on success, otherwise None.
    """
    access_token = _extract_access_token()
    if not access_token:
        return None

    supabase = current_app.supabase
    try:
        return supabase.auth.get_user(access_token).user
    except Exception:
        return None