import torch
import torch.nn.functional as nnf
from torch import nn
from transformers import AutoModelForCausalLM
from meshanything_train.miche.encode import load_model
from meshanything_train.models.shape_opt import ShapeOPTConfig

from einops import repeat, reduce, rearrange, pack, unpack

def coor_discretize(
    t,
    continuous_range, # (-0.5, 0.5)
    num_discrete: int = 128
):
    lo, hi = continuous_range
    assert hi > lo

    t = (t - lo) / (hi - lo) # 0 <=t < 1
    t *= num_discrete # [0, num_discrete-1]
    assert (t - t.round()).sum() == 0
    assert (t <= num_discrete-1).all() and (t >= 0).all()  # 0 to num_discrete-1

    return t.long()

def undiscretize(
    t,
    low,#-0.5
    high,# 0.5
    num_discrete
):
    assert (t >= 0).all() and (t <= num_discrete-1).all()
    assert high>low
    t = t.float() #[0, num_discrete-1]

    t /= num_discrete  # 0<=t<1
    t = t * (high - low) + low # -0.5 <= t < 0.5
    assert (t < high).all() and (t >= low).all()
    return t

class SingleGPT(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.point_encoder = load_model()
        self.cond_length = 257
        self.cond_dim = 768

        self.n_discrete_size = args.n_discrete_size
        self.max_seq_ratio = self.args.max_seq_ratio
        self.face_per_token = 9
        self.pad_id = -1
        self.max_vertices = args.max_vertices

        self.max_length = int(args.n_max_triangles * self.face_per_token * self.max_seq_ratio + 3 + self.cond_length)
        self.gen_max_length = int(args.gen_n_max_triangles * self.face_per_token * self.max_seq_ratio + 3 + self.cond_length)

        self.coor_continuous_range = (-0.5, 0.5)

        vocab_size = self.n_discrete_size + 4 # 4 for bos, eos, pad, &
        self.config = ShapeOPTConfig.from_pretrained(
            args.llm,
            n_positions=self.max_length,
            max_position_embeddings=self.max_length,
            vocab_size = vocab_size,
            _attn_implementation="flash_attention_2"
        )

        self.bos_token_id = 0
        self.eos_token_id = 1
        self.pad_token_id = 2

        self.config.bos_token_id = self.bos_token_id
        self.config.eos_token_id = self.eos_token_id
        self.config.pad_token_id = self.pad_token_id
        self.config._attn_implementation ="flash_attention_2"
        self.config.n_discrete_size = self.n_discrete_size
        self.config.face_per_token = self.face_per_token
        self.config.cond_length = self.cond_length
        self.config.max_vertices = args.max_vertices
        self.config.word_embed_proj_dim = self.config.hidden_size

        self.transformer = AutoModelForCausalLM.from_config(
            config=self.config, use_flash_attention_2 = True
        )

        self.cond_head_proj = nn.Linear(self.cond_dim, self.config.word_embed_proj_dim)
        self.cond_proj = nn.Linear(self.cond_dim * 2, self.config.word_embed_proj_dim)

        self.train()

    def loop_detokenize(self, input_ids):
        input_ids = input_ids.reshape(input_ids.shape[0], -1) # B x L
        batch_size = input_ids.shape[0]
        continuous_coors = torch.zeros((batch_size, self.args.n_max_triangles * 3 * 10, 3), device=input_ids.device)
        continuous_coors[...] = float('nan')
        for i in range(batch_size):
            cur_ids = input_ids[i]
            coor_loop_check = 0
            vertice_count = 0
            continuous_coors[i, :3, :] = torch.tensor([[-0.1, 0.0, 0.1], [-0.1, 0.1, 0.2], [-0.3, 0.3, 0.2]],
                                                      device=input_ids.device)
            error_judge = 0
            for id in cur_ids:
                if id == self.pad_id:
                    if coor_loop_check < 9:
                        error_judge=1
                    if coor_loop_check % 3 != 0:
                        error_judge=1
                    break
                elif id == self.n_discrete_size:
                    if coor_loop_check < 9:
                        error_judge=1
                        break
                    if coor_loop_check % 3 !=0:
                        error_judge=1
                        break
                    coor_loop_check = 0
                else:

                    if coor_loop_check % 3 == 0 and coor_loop_check >= 9:
                        continuous_coors[i, vertice_count] = continuous_coors[i, vertice_count-2]
                        continuous_coors[i, vertice_count+1] = continuous_coors[i, vertice_count-1]
                        vertice_count += 2
                    continuous_coors[i, vertice_count, coor_loop_check % 3] = undiscretize(id, self.coor_continuous_range[0], self.coor_continuous_range[1], self.n_discrete_size)
                    if coor_loop_check % 3 == 2:
                        vertice_count += 1
                    coor_loop_check += 1
            if vertice_count <= 3:
                error_judge=1

            if coor_loop_check % 3 != 0:
                error_judge=1

            if error_judge:
                continuous_coors[i, -1, -1] = 0

        continuous_coors = rearrange(continuous_coors, 'b (nf nv) c -> b nf nv c', nv=3, c=3)

        return continuous_coors # b, nf, 3, 3

    def train(self, mode: bool = True):
        super().train(mode)
        if hasattr(self,"point_encoder"):
            self.point_encoder.eval()
            for param in self.point_encoder.parameters():
                param.requires_grad = False

    def forward(self, data_dict: dict, is_eval: bool = False) -> dict:
        if not is_eval:
            return self.train_one_step(data_dict)
        else:
            return self.generate(data_dict)

    def pad_id_and_attn(self, input_ids, attention_mask, face_ids = None): # same
        # reserve one space for `bos`, the pad_id will be replaced to `bos`
        place_holder = torch.ones_like(input_ids[:, [0]])   # batch x 1
        # prepare input_ids and attention_mask for transformers
        input_ids[attention_mask.bool()] += 3 # 0 - num_tokens to 3 - num_tokens + 3, total: 0 - num_tokens + 3, num: numtokens + 4
        input_ids[~attention_mask.bool()] = self.pad_token_id # in transformers pad token id is only used for init nn.embedding which we won't use
        if face_ids is None:
            face_ids = repeat(torch.arange(0, self.face_per_token, device=input_ids.device), 'f -> b (k f)', b=input_ids.shape[0], k=input_ids.shape[1]//self.face_per_token) + 3

        input_ids = torch.cat(
            (place_holder * self.bos_token_id, input_ids, place_holder * self.pad_token_id),
            dim=1
        )
        input_ids[torch.arange(0, input_ids.shape[0]), attention_mask.sum(dim=1).long()+1] = self.eos_token_id

        face_ids[~attention_mask.bool()] = self.pad_token_id
        face_ids = torch.cat(
            (place_holder * self.bos_token_id, face_ids, place_holder * self.pad_token_id),
            dim=1
        )
        face_ids[torch.arange(0, face_ids.shape[0]), attention_mask.sum(dim=1).long()+1] = self.eos_token_id

        attention_mask = torch.cat(
            (place_holder, place_holder, attention_mask, ),
            dim=1
        )
        # length
        return input_ids, face_ids, attention_mask

    def process_point_feature(self, point_feature):
        encode_feature = torch.zeros(self.args.batchsize_per_gpu, self.cond_length, self.config.word_embed_proj_dim,
                                    device=self.cond_head_proj.weight.device, dtype=self.cond_head_proj.weight.dtype)
        encode_feature[:, 0] = self.cond_head_proj(point_feature[:, 0])
        shape_latents = self.point_encoder.to_shape_latents(point_feature[:, 1:])
        encode_feature[:, 1:] = self.cond_proj(torch.cat([point_feature[:, 1:], shape_latents], dim=-1))

        return encode_feature

    def train_one_step(self, data_dict: dict) -> dict:
        point_feature = self.point_encoder.encode_latents(data_dict["pc_normal"])
        with torch.no_grad():
            assert "sequence" in data_dict
            input_ids = data_dict['sequence']
            face_ids = data_dict['id_sequence']
            attention_mask = input_ids != self.pad_id

            sequence_max_length = attention_mask.sum(dim=1).max()

            input_ids = input_ids[:, :sequence_max_length]
            attention_mask = attention_mask[:, :sequence_max_length]
            face_ids = face_ids[:, :sequence_max_length]

            # detokenize_check = self.loop_detokenize(input_ids)
            # assert torch.all(data_dict['faces']!=-1,dim=-1).sum() == torch.all(torch.all(detokenize_check == detokenize_check, dim=-1),dim=-1).sum()
            input_ids, face_ids, attention_mask = self.pad_id_and_attn(input_ids, attention_mask, face_ids=face_ids)

        # add cond_length to attention mask
        pad_attention_mask = torch.ones((attention_mask.shape[0], self.cond_length), device=attention_mask.device, dtype=attention_mask.dtype)
        attention_mask = torch.concatenate((pad_attention_mask, attention_mask), dim=1)

        processed_point_feature = self.process_point_feature(point_feature=point_feature)

        output = self.transformer(
            inputs_embeds = processed_point_feature,
            input_ids=input_ids,
            face_ids=face_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        # compute loss with shift token right
        logit = output.logits[:, self.cond_length-1:-1]  # batch x ntoken x vocab
        label = input_ids[:, 0:]  # batch x ntoken
        masks = attention_mask[:, self.cond_length-1:-1]  # batch x ntoken
        # also predict bos token
        loss_per_token = nnf.cross_entropy(
            logit.permute(0, 2, 1),  # batch x vocab x ntoken
            label,
            reduction='none'
        )  # batch x ntoken
        final_loss = torch.sum(loss_per_token * masks) / (torch.sum(masks) + 1e-8)
        data_dict['loss'] = final_loss

        return data_dict

    @torch.no_grad()
    def generate(self, data_dict) -> dict:

        point_feature = self.point_encoder.encode_latents(data_dict["pc_normal"])
        processed_point_feature = self.process_point_feature(point_feature)
        generate_length = self.gen_max_length - self.cond_length
        net_device = next(self.parameters()).device
        outputs = torch.ones(self.args.batchsize_per_gpu, generate_length).long().to(net_device) * self.eos_token_id
        # batch x ntokens
        if self.args.num_beams is not None and "pc_normal" in data_dict:
            results = self.transformer.generate(
                inputs_embeds=processed_point_feature,
                max_new_tokens=generate_length,  # all faces plus two
                num_beams=self.args.num_beams,
                bos_token_id=self.bos_token_id,
                eos_token_id=self.eos_token_id,
                pad_token_id=self.pad_token_id,
            )
        else:
            results = self.transformer.generate(
                inputs_embeds = processed_point_feature,
                max_new_tokens = generate_length, # all faces plus two
                do_sample=True,
                top_k=50,
                top_p=0.95,
                bos_token_id = self.bos_token_id,
                eos_token_id = self.eos_token_id,
                pad_token_id = self.pad_token_id,
            )
        assert results.shape[1] <= generate_length # B x ID  bos is not included since it's predicted
        outputs[:, :results.shape[1]] = results
        # batch x ntokens ====> batch x ntokens x D
        outputs = outputs[:, 1: -1] # eos and bos removed

        outputs[outputs == self.bos_token_id] = self.pad_id
        outputs[outputs == self.eos_token_id] = self.pad_id
        outputs[outputs == self.pad_token_id] = self.pad_id

        outputs[outputs != self.pad_id] -= 3
        gen_mesh = self.loop_detokenize(outputs)


        return gen_mesh

