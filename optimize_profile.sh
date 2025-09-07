#!/bin/bash

# Profile Image Optimization Script
# This script converts your profile image to optimized WebP format for faster loading

echo "🖼️  Profile Image Optimization Script"
echo "======================================"

# Check if ImageMagick is installed
if ! command -v convert &> /dev/null; then
    echo "❌ ImageMagick not found. Installing..."

    # Detect OS and install ImageMagick
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v brew &> /dev/null; then
            brew install imagemagick
        else
            echo "Please install Homebrew first: https://brew.sh"
            exit 1
        fi
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        sudo apt-get update && sudo apt-get install -y imagemagick
    else
        echo "Please install ImageMagick manually for your OS"
        exit 1
    fi
fi

# Check if WebP tools are available
if ! command -v cwebp &> /dev/null; then
    echo "❌ WebP tools not found. Installing..."

    if [[ "$OSTYPE" == "darwin"* ]]; then
        brew install webp
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo apt-get install -y webp
    fi
fi

# Navigate to content directory
cd "$(dirname "$0")/content" || exit 1

# Check if profile.PNG exists
if [ ! -f "profile.PNG" ]; then
    echo "❌ profile.PNG not found in content directory"
    echo "Please ensure your profile image is named 'profile.PNG'"
    exit 1
fi

echo "📊 Original file info:"
ls -lh profile.PNG

# Get original file size
original_size=$(stat -f%z profile.PNG 2>/dev/null || stat -c%s profile.PNG 2>/dev/null)

echo ""
echo "🔄 Converting to optimized formats..."

# Create optimized WebP version (high quality, smaller size)
cwebp -q 85 -resize 360 360 profile.PNG -o profile.webp

# Create optimized PNG fallback (smaller than original)
convert profile.PNG -resize 360x360^ -gravity center -extent 360x360 -quality 85 profile_optimized.PNG

# Replace original with optimized version
mv profile_optimized.PNG profile.PNG

echo ""
echo "✅ Optimization complete!"

# Show results
if [ -f "profile.webp" ]; then
    webp_size=$(stat -f%z profile.webp 2>/dev/null || stat -c%s profile.webp 2>/dev/null)
    echo "📊 WebP size: $(numfmt --to=iec $webp_size)"
fi

new_png_size=$(stat -f%z profile.PNG 2>/dev/null || stat -c%s profile.PNG 2>/dev/null)
echo "📊 Optimized PNG size: $(numfmt --to=iec $new_png_size)"

# Calculate savings
if [ -n "$original_size" ] && [ -n "$new_png_size" ]; then
    savings=$((original_size - new_png_size))
    percentage=$((savings * 100 / original_size))
    echo "💾 Space saved: $(numfmt --to=iec $savings) (${percentage}%)"
fi

echo ""
echo "🎉 Your profile image is now optimized!"
echo "📝 The website will automatically use:"
echo "   • WebP format for modern browsers (fastest)"
echo "   • Optimized PNG as fallback for older browsers"
echo ""
echo "⚡ Expected improvements:"
echo "   • 60-80% smaller file size"
echo "   • Faster loading times"
echo "   • Better user experience"

# Optional: Create different sizes for responsive design
echo ""
read -p "🤔 Create additional responsive sizes? (y/n): " create_responsive

if [[ $create_responsive == "y" || $create_responsive == "Y" ]]; then
    echo "🔄 Creating responsive sizes..."

    # Small mobile (120px)
    cwebp -q 85 -resize 240 240 profile.PNG -o profile-small.webp
    convert profile.PNG -resize 240x240^ -gravity center -extent 240x240 -quality 85 profile-small.PNG

    # Medium mobile/tablet (150px display = 300px actual)
    cwebp -q 85 -resize 300 300 profile.PNG -o profile-medium.webp
    convert profile.PNG -resize 300x300^ -gravity center -extent 300x300 -quality 85 profile-medium.PNG

    echo "✅ Responsive sizes created!"
    echo "📱 Available sizes:"
    echo "   • profile-small.webp/PNG (240px - for mobile)"
    echo "   • profile-medium.webp/PNG (300px - for tablets)"
    echo "   • profile.webp/PNG (360px - for desktop)"
fi

echo ""
echo "🚀 Your profile image optimization is complete!"
echo "💡 Tip: Test your website loading speed before and after to see the improvement!"
