from flask import Blueprint, request, jsonify, current_app
from datetime import datetime

sys_prompt_bp = Blueprint('prompt', __name__)

def authenticate_user(access_token):
    supabase = current_app.supabase
    try:
        user_response = supabase.auth.get_user(access_token)
        return user_response.user
    except:
        return None

@sys_prompt_bp.route('/update_user_system_prompt', methods=['POST'])
def update_user_system_prompt():
    data = request.get_json()
    access_token = data.get('access_token')
    new_prompt = data.get('system_prompt')

    if not all([access_token, new_prompt]):
        return jsonify({'error': 'access_token and system_prompt required'}), 400

    user = authenticate_user(access_token)
    if not user:
        return jsonify({'error': 'Invalid token'}), 401

    supabase = current_app.supabase

    # Upsert user's custom prompt
    resp = supabase.table('user_system_prompts').upsert({
        'user_id': user.id,
        'system_prompt': new_prompt,
        'updated_at': datetime.utcnow().isoformat()
    }).execute()

    if resp.data:
        return jsonify({'message': 'User system prompt updated successfully'}), 200
    else:
        return jsonify({'error': 'Update failed'}), 400

@sys_prompt_bp.route('/get_user_system_prompt', methods=['GET'])
def get_user_system_prompt():
    access_token = request.args.get('access_token')

    if not access_token:
        return jsonify({'error': 'access_token required'}), 400

    user = authenticate_user(access_token)
    if not user:
        return jsonify({'error': 'Invalid token'}), 401

    supabase = current_app.supabase

    # Check user's existing prompt explicitly
    prompt_resp = supabase.table('user_system_prompts') \
                          .select('system_prompt') \
                          .eq('user_id', user.id) \
                          .execute()

    if prompt_resp.data and len(prompt_resp.data) > 0:
        return jsonify({'system_prompt': prompt_resp.data[0]['system_prompt']}), 200

    # Explicitly check global default prompt
    global_resp = supabase.table('global_system_prompt') \
                          .select('config_value') \
                          .eq('config_key', 'system_prompt') \
                          .execute()

    if global_resp.data and len(global_resp.data) > 0:
        default_prompt = global_resp.data[0]['config_value']

        # Insert default prompt for user explicitly
        supabase.table('user_system_prompts').insert({
            'user_id': user.id,
            'system_prompt': default_prompt
        }).execute()

        return jsonify({'system_prompt': default_prompt}), 200

    # Clear error handling:
    return jsonify({'error': 'Global default prompt missing or misconfigured'}), 500
