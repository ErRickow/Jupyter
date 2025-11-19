#!/bin/bash
# Quick deployment script untuk Baseten

echo "=================================================="
echo "🚀 Unsloth TTS - Baseten Deployment Script"
echo "=================================================="
echo ""

# Check if truss is installed
if ! command -v truss &> /dev/null; then
    echo "❌ Truss CLI not found!"
    echo "📦 Installing truss..."
    pip install truss
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install truss"
        exit 1
    fi
    echo "✅ Truss installed successfully"
fi

echo ""
echo "🔐 Checking Baseten authentication..."

# Check if logged in (truss will prompt if not)
truss whoami &> /dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Not logged in to Baseten"
    echo "🔑 Running: truss login"
    truss login
    if [ $? -ne 0 ]; then
        echo "❌ Login failed"
        exit 1
    fi
fi

echo "✅ Authenticated with Baseten"
echo ""

# Confirm deployment
echo "📋 Deployment Configuration:"
echo "   Model: Unsloth TTS"
echo "   GPU: T4"
echo "   Location: $(pwd)"
echo ""

read -p "🚀 Deploy to Baseten? (y/n): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Deployment cancelled"
    exit 0
fi

echo ""
echo "📤 Pushing model to Baseten..."
echo "⏳ This may take 5-10 minutes..."
echo ""

# Push to Baseten
truss push

if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "✅ Deployment successful!"
    echo "=================================================="
    echo ""
    echo "📝 Next steps:"
    echo "   1. Copy your Model URL and API Key from Baseten dashboard"
    echo "   2. Test with: python test_inference.py"
    echo ""
    echo "🔗 Dashboard: https://app.baseten.co/"
    echo ""
else
    echo ""
    echo "❌ Deployment failed"
    echo "💡 Check the error messages above"
    exit 1
fi
