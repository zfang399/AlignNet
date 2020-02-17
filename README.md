# AlignNet
AlignNet: A Unifying Approach to Audio-Visual Alignment

Example training command:
<pre>
python3 train.py --mode train --local_distortion --global_stretch --global_shift --experiment_name <i>YOUR_EXPERIMENT_NAME</i> --log_dir <i>YOUR_LOG_DIRECTORY</i>
</pre>
Remove the flags `--local_distortion`, `--local_distortion`, `--local_distortion` if we want to keep only some of the three manipulations.
