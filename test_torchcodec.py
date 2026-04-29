from torchcodec.decoders import VideoDecoder
import os
video_root = "/mnt/pvc/training_data/lerobot_v21_ort6d/taco_play/videos/chunk-001/observation.images.rgb_static"
video_names = sorted(os.listdir(video_root))
for video_name in video_names:
    video_path = os.path.join(video_root, video_name)
    decoder = VideoDecoder(video_path, seek_mode="approximate")
    print(decoder)