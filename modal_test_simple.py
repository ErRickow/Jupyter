"""
Simple Modal Test - Check Authentication
"""
import os
import modal

# Set credentials
os.environ['MODAL_TOKEN_ID'] = 'ak-2fPXyE2kShNW8RBPaLamY7'
os.environ['MODAL_TOKEN_SECRET'] = 'as-FKBZ1MK1id05l6BaBNiN6x'

print("Testing Modal Authentication...")
print(f"Modal version: {modal.__version__}")

# Create simple app
app = modal.App("test-simple")

@app.function()
def hello():
    return "Hello from Modal!"

if __name__ == "__main__":
    print("\n✅ App created successfully")
    print(f"App name: {app.name}")
    print("\nTo deploy: modal deploy modal_test_simple.py")
    print("To run: modal run modal_test_simple.py")
