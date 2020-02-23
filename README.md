# AlignNet
This is the codebase for "AlignNet: A Unifying Approach to Audio-Visual Alignment", WACV 2020 <br/>
Website: <url>https://jianrenw.github.io/AlignNet/</url>

### Training:
<pre>
python3 train.py --mode train --local_distortion --global_stretch --global_shift --experiment_name <i>YOUR_EXPERIMENT_NAME</i> --log_dir <i>YOUR_LOG_DIRECTORY</i>
</pre>
Remove the flags `--local_distortion`, `--local_distortion`, `--local_distortion` if you want to keep only some of the three manipulations.
