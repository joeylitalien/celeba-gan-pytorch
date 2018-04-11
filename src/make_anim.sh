ffmpeg -f image2 -i ../interpolated/test%d.png ../interpolated/out.mp4
ffmpeg -i ../interpolated/out.mp4 -pix_fmt rgb24 ../interpolated/out.gif
