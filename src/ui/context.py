def get_user_initials(user):
    first = (user.first_name or "").strip()
    last = (user.last_name or "").strip()
    username = (user.username or "").strip()

    if first and last:
        return f"{first[0]}{last[0]}".upper()
    if first:
        return first[0].upper()
    if username:
        return username[:2].upper()

    return "??"

def user_ui_context(request):
    if request.user.is_authenticated:
        return {
            "user_initials": get_user_initials(request.user),
        }

    return {
        "user_initials": "",
    }