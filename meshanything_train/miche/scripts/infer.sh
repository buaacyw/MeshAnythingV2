python inference.py \
--task reconstruction \
--config_path ./configs/aligned_shape_latents/shapevae-256.yaml \
--ckpt_path ./checkpoints/aligned_shape_latents/shapevae-256.ckpt \
--pointcloud_path ./example_data/surface/surface.npz

python inference.py \
--task image2mesh \
--config_path ./configs/image_cond_diffuser_asl/image-ASLDM-256.yaml \
--ckpt_path ./checkpoints/image_cond_diffuser_asl/image-ASLDM-256.ckpt \
--image_path ./example_data/image/car.jpg

python inference.py \
--task text2mesh \
--config_path ./configs/text_cond_diffuser_asl/text-ASLDM-256.yaml \
--ckpt_path ./checkpoints/text_cond_diffuser_asl/text-ASLDM-256.ckpt \
--text "A 3D model of motorcar; Porche Cayenne Turbo."