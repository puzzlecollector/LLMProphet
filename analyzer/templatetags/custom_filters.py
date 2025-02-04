from django import template

register = template.Library()

@register.filter(name='split')
def split(value, arg):
    """Splits a string into a list using the given delimiter."""
    return value.split(arg)

@register.filter(name='concat')
def concat(value, arg):
    """Concatenates two strings."""
    return str(value) + str(arg)

@register.filter(name='get_dict_value')
def get_dict_value(dictionary, key):
    """Safely retrieves a value from a dictionary using a key."""
    if isinstance(dictionary, dict):
        return dictionary.get(key, 0)  # Default to 0 if key doesn't exist
    return 0

@register.filter(name='to_float')
def to_float(value):
    """Converts a value to float."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0  # Default to 0.0 if conversion fails
