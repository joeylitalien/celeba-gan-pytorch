# Check ffmpeg is installed
ffmpeg=$(dpkg-query -W -f='${Status}' ffmpeg | grep -c "ok installed")

# Nope, install it
if [ $ffmpeg -eq 0 ]; then
  echo "Missing ffmpeg, installing..."
  sudo apt-get update
  sudo apt-get install ffmpeg
fi

# Ready to generate video, takes frame directory as input
echo "Creating video..."
ffmpeg -f image2 -i $1/test%d.png $1/anim.mp4
ffmpeg -i $1/anim.mp4 -pix_fmt rgb24 $1/anim.gif

# Old code
# ffmpeg -f image2 -i ../interpolated/test%d.png ../interpolated/anim.mp4
# ffmpeg -i ../interpolated/anim.mp4 -pix_fmt rgb24 ../interpolated/anim.gif
