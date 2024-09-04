import torch
from torch import nn
from transformers import AutoModelForCausalLM
import importlib

from einops import rearrange
from huggingface_hub import PyTorchModelHubMixin

from ...MeshAnything.miche.encode import load_model
from ...MeshAnything.models.shape_opt import ShapeOPTConfig

class MeshAnythingV2(
    nn.Module,
    PyTorchModelHubMixin,
    repo_url="https://github.com/buaacyw/MeshAnythingV2",
    pipeline_tag="image-to-3d",
    license="mit",
):
    def __init__(self, config={}):
        super().__init__()

        self.config = config
        self.point_encoder = load_model(ckpt_path=None)
        self.n_discrete_size = 128
        self.max_seq_ratio = 0.70
        self.face_per_token = 9
        self.cond_length = 257
        self.cond_dim = 768
        self.pad_id = -1
        self.n_max_triangles = 1600
        self.max_length = int(
            self.n_max_triangles * self.face_per_token * self.max_seq_ratio
            + 3
            + self.cond_length
        )  # add 1

        self.coor_continuous_range = (-0.5, 0.5)

        self.config = ShapeOPTConfig.from_pretrained(
            "facebook/opt-350m",
            n_positions=self.max_length,
            max_position_embeddings=self.max_length,
            vocab_size=self.n_discrete_size + 4,
            _attn_implementation="flash_attention_2",
        )

        self.bos_token_id = 0
        self.eos_token_id = 1
        self.pad_token_id = 2

        self.config.bos_token_id = self.bos_token_id
        self.config.eos_token_id = self.eos_token_id
        self.config.pad_token_id = self.pad_token_id
        self.config._attn_implementation = "flash_attention_2"
        self.config.n_discrete_size = self.n_discrete_size
        self.config.face_per_token = self.face_per_token
        self.config.cond_length = self.cond_length

        if self.config.word_embed_proj_dim != self.config.hidden_size:
            self.config.word_embed_proj_dim = self.config.hidden_size
        self.transformer = AutoModelForCausalLM.from_config(
            config=self.config, use_flash_attention_2=True
        )
        self.transformer.to_bettertransformer()

        self.cond_head_proj = nn.Linear(self.cond_dim, self.config.word_embed_proj_dim)
        self.cond_proj = nn.Linear(self.cond_dim * 2, self.config.word_embed_proj_dim)

        self.eval()

    def adjacent_detokenize(self, input_ids):
        input_ids = input_ids.reshape(input_ids.shape[0], -1)  # B x L
        batch_size = input_ids.shape[0]
        continuous_coors = torch.zeros(
            (batch_size, self.n_max_triangles * 3 * 10, 3), device=input_ids.device
        )
        continuous_coors[...] = float("nan")

        for i in range(batch_size):
            cur_ids = input_ids[i]
            coor_loop_check = 0
            vertice_count = 0
            continuous_coors[i, :3, :] = torch.tensor(
                [[-0.1, 0.0, 0.1], [-0.1, 0.1, 0.2], [-0.3, 0.3, 0.2]],
                device=input_ids.device,
            )
            for id in cur_ids:
                if id == self.pad_id:
                    break
                elif id == self.n_discrete_size:
                    if coor_loop_check < 9:
                        break
                    if coor_loop_check % 3 != 0:
                        break
                    coor_loop_check = 0
                else:

                    if coor_loop_check % 3 == 0 and coor_loop_check >= 9:
                        continuous_coors[i, vertice_count] = continuous_coors[
                            i, vertice_count - 2
                        ]
                        continuous_coors[i, vertice_count + 1] = continuous_coors[
                            i, vertice_count - 1
                        ]
                        vertice_count += 2
                    continuous_coors[i, vertice_count, coor_loop_check % 3] = (
                        undiscretize(
                            id,
                            self.coor_continuous_range[0],
                            self.coor_continuous_range[1],
                            self.n_discrete_size,
                        )
                    )
                    if coor_loop_check % 3 == 2:
                        vertice_count += 1
                    coor_loop_check += 1

        continuous_coors = rearrange(
            continuous_coors, "b (nf nv) c -> b nf nv c", nv=3, c=3
        )

        return continuous_coors  # b, nf, 3, 3

    def forward(self, data_dict: dict, is_eval: bool = False) -> dict:
        if not is_eval:
            return self.train_one_step(data_dict)
        else:
            return self.generate(data_dict)

    def process_point_feature(self, point_feature):
        encode_feature = torch.zeros(
            point_feature.shape[0],
            self.cond_length,
            self.config.word_embed_proj_dim,
            device=self.cond_head_proj.weight.device,
            dtype=self.cond_head_proj.weight.dtype,
        )
        encode_feature[:, 0] = self.cond_head_proj(point_feature[:, 0])
        shape_latents = self.point_encoder.to_shape_latents(point_feature[:, 1:])
        encode_feature[:, 1:] = self.cond_proj(
            torch.cat([point_feature[:, 1:], shape_latents], dim=-1)
        )

        return encode_feature

    @torch.no_grad()
    def forward(self, pc_normal, sampling=False) -> dict:
        batch_size = pc_normal.shape[0]
        point_feature = self.point_encoder.encode_latents(pc_normal)
        processed_point_feature = self.process_point_feature(point_feature)
        generate_length = self.max_length - self.cond_length
        net_device = next(self.parameters()).device
        outputs = (
            torch.ones(batch_size, generate_length).long().to(net_device)
            * self.eos_token_id
        )
        # batch x ntokens
        if not sampling:
            results = self.transformer.generate(
                inputs_embeds=processed_point_feature,
                max_new_tokens=generate_length,  # all faces plus two
                num_beams=1,
                bos_token_id=self.bos_token_id,
                eos_token_id=self.eos_token_id,
                pad_token_id=self.pad_token_id,
            )
        else:
            results = self.transformer.generate(
                inputs_embeds=processed_point_feature,
                max_new_tokens=generate_length,  # all faces plus two
                do_sample=True,
                top_k=50,
                top_p=0.95,
                bos_token_id=self.bos_token_id,
                eos_token_id=self.eos_token_id,
                pad_token_id=self.pad_token_id,
            )
        assert (
            results.shape[1] <= generate_length
        )  # B x ID  bos is not included since it's predicted
        outputs[:, : results.shape[1]] = results
        # batch x ntokens ====> batch x ntokens x D
        outputs = outputs[:, 1:-1]

        outputs[outputs == self.bos_token_id] = self.pad_id
        outputs[outputs == self.eos_token_id] = self.pad_id
        outputs[outputs == self.pad_token_id] = self.pad_id

        outputs[outputs != self.pad_id] -= 3
        gen_mesh = self.adjacent_detokenize(outputs)

        return gen_mesh


def undiscretize(t, low, high, num_discrete):  # -0.5  # 0.5
    t = t.float()  # [0, num_discrete-1]

    t /= num_discrete  # 0<=t<1
    t = t * (high - low) + low  # -0.5 <= t < 0.5
    return t
