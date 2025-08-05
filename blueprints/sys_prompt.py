from flask import Blueprint, request, jsonify, current_app
from datetime import datetime

sys_prompt_bp = Blueprint('prompt', __name__)

# ---------- helpers ---------------------------------------------------------

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

# ---------- routes ----------------------------------------------------------

@sys_prompt_bp.route('/update_user_system_prompt', methods=['POST'])
def update_user_system_prompt():
    """Upsert a custom system prompt for the authenticated user."""
    body = request.get_json(silent=True) or {}
    new_prompt = body.get('system_prompt')

    if not new_prompt:
        return jsonify({'error': 'system_prompt is required'}), 400

    user = authenticate_user()
    if not user:
        return jsonify({'error': 'Invalid or missing token'}), 401

    supabase = current_app.supabase
    resp = (
        supabase.table('user_system_prompts')
        .upsert(
            {
                'user_id': user.id,
                'system_prompt': new_prompt,
                'updated_at': datetime.utcnow().isoformat(),
            }
        )
        .execute()
    )

    if resp.data:
        return jsonify({'message': 'User system prompt updated successfully'}), 200
    return jsonify({'error': 'Update failed'}), 400


@sys_prompt_bp.route('/get_user_system_prompt', methods=['GET'])
def get_user_system_prompt():
    """Return the user-specific system prompt, or fall back to the global one."""
    user = authenticate_user()
    if not user:
        return jsonify({'error': 'Invalid or missing token'}), 401

    supabase = current_app.supabase

    # 1⃣ Check if the user already has a prompt
    prompt_resp = (
        supabase.table('user_system_prompts')
        .select('system_prompt')
        .eq('user_id', user.id)
        .execute()
    )

    if prompt_resp.data:
        return jsonify({'system_prompt': prompt_resp.data[0]['system_prompt']}), 200

    # 2⃣ Otherwise get the global default and cache it for the user
    global_resp = (
        supabase.table('global_system_prompt')
        .select('config_value')
        .eq('config_key', 'system_prompt')
        .execute()
    )

    if global_resp.data:
        default_prompt = global_resp.data[0]['config_value']
        supabase.table('user_system_prompts').insert(
            {'user_id': user.id, 'system_prompt': default_prompt}
        ).execute()
        return jsonify({'system_prompt': default_prompt}), 200

    return jsonify({'error': 'Global default prompt missing or misconfigured'}), 500
