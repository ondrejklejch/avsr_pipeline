#!/bin/bash

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

. path.sh

stage=2
list="id_selected_talks.txt"
dir="av2022/eval"
resampled_dir="av2022/eval_25fps"
output_dir="av2022/eval_output_with_full_video"

mkdir -p $dir $resampled_dir $output_dir

if [ $stage -le 0 ]; then
  yt-dlp \
    -4 --socket-timeout 1 -ciw \
    -f "b[height=720][ext=mp4]" \
    --all-subs \
    --output "${dir}/%(id)s.%(ext)s" \
    -v --match-filter "!is_live" \
    -a ${list}
fi

if [ $stage -le 1 ]; then
  for f in $dir/*.mp4; do
    name=`basename $f`
    echo $name
    ffmpeg -y -i $f -filter:v fps=25 -strict -2 $resampled_dir/$name
  done
fi

if [ $stage -le 2 ]; then
  find $resampled_dir -name "*.mp4" > $resampled_dir/input_files
  split -n l/4 --numeric-suffixes=1 $resampled_dir/{input_files,input_files.}

  for i in 1 2 3 4; do
    python3 -u run_eval_with_full_video.py --device cuda:$((i-1)) $resampled_dir/input_files.0${i} $output_dir | tee $output_dir/log.$i &
  done

  wait
fi
