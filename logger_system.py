import functools
import logging
import sys

# Configure basic logging to console
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def log_wrapper(func):
    """Universal decorator for function logging."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Entry Log
        params = f"args={args}, kwargs={kwargs}"
        print(f"🚀 Executing: [{func.__name__}] | Params: {params}")
        
        try:
            result = func(*args, **kwargs)
            # Exit/Success Log
            print(f"✅ Success: [{func.__name__}] | Result: {type(result).__name__}")
            return result
        except Exception as e:
            # Error Log
            print(f"❌ FAULT DETECTED in [{func.__name__}]: {str(e)}")
            raise e
            
    return wrapper

def log_action(action_name, details=""):
    """Helper for non-function events like buttons."""
    print(f"🔘 Button Triggered: [{action_name}] | Details: {details}")
