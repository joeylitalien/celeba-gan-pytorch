#!/bin/sh

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
mkdir $1/video
ffmpeg -y -f image2 -i $1/frame%d.png $1/video/anim.mp4
ffmpeg -y -i $1/video/anim.mp4 -pix_fmt rgb24 $1/video/anim.gif

# Clean extra frames
ls $1/frame* | sort --version-sort | tail -$2 | xargs rm
