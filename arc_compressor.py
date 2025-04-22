import numpy as np
import torch
import initializers
import layers
import torch.nn as nn
import multitensor_systems

np.random.seed(0)
torch.manual_seed(0)
torch.set_default_dtype(torch.float32)
torch.set_default_device('cuda')

def add_noise_multitensor(x, sigma):
    return multitensor_systems.multify(
        lambda dims, t: t + sigma * torch.randn_like(t)
    )(x)

class ARCCompressor:
    """
    VAE Decoder with on-the-fly color permutation augmentation.
    """
    n_layers = 2
    share_up_dim = 64
    share_down_dim = 32
    decoding_dim = 16
    softmax_dim = 8
    cummax_dim = 16
    shift_dim = 16
    nonlinear_dim = 512

    def channel_dim_fn(self, dims):
        return 16 if dims[2] == 0 else 8

    def __init__(self, task):
        self.task = task
        # Store original ground-truth tensor for color-aug
        self.problem_orig = task.problem.clone()
        self.n_colors_plus1 = task.n_colors + 1

        self.multitensor_system = task.multitensor_system
        initializer = initializers.Initializer(self.multitensor_system, self.channel_dim_fn)

        self.multiposteriors = initializer.initialize_multiposterior(self.decoding_dim)
        self.decode_weights = initializer.initialize_multilinear([self.decoding_dim, self.channel_dim_fn])
        initializer.symmetrize_xy(self.decode_weights)
        self.target_capacities = initializer.initialize_multizeros([self.decoding_dim])

        # Attention, GRU, SE, and other modules (unchanged)
        embed_dim = self.channel_dim_fn([1,1,0,1,1])
        self.spatial_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8, batch_first=False)
        self.global_attn  = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8, batch_first=True)
        E = embed_dim
        self.temporal_attn = nn.MultiheadAttention(embed_dim=E, num_heads=8, batch_first=True)
        self.obj_q   = nn.Linear(E, E, bias=False)
        self.obj_k   = nn.Linear(E, E, bias=False)
        self.obj_v   = nn.Linear(E, E, bias=False)
        self.obj_out = nn.Linear(E, E, bias=False)

        class _ConvGRU(nn.Module):
            def __init__(self, E):
                super().__init__()
                self.E = E
                self.conv_zr = nn.Conv2d(E*2, E*2, 3, padding=1)
                self.conv_h  = nn.Conv2d(E*2, E,   3, padding=1)
            def forward(self, h, x):
                z, r = torch.split(self.conv_zr(torch.cat([h,x],1)), self.E, 1)
                z, r = torch.sigmoid(z), torch.sigmoid(r)
                h_hat = torch.tanh(self.conv_h(torch.cat([r*h, x],1)))
                return (1-z)*h + z*h_hat
        self.gru = _ConvGRU(E)

        self.ca_q = nn.Conv2d(E, E, 1, bias=False)
        self.ca_k = nn.Linear(E, E, bias=False)
        self.ca_v = nn.Linear(E, E, bias=False)

        C = embed_dim
        self.se_fc1 = nn.Linear(C, C//4)
        self.se_fc2 = nn.Linear(C//4, C)

        # MultiTensor-system weights
        self.share_up_weights = []
        self.share_down_weights = []
        self.softmax_weights = []
        self.cummax_weights  = []
        self.shift_weights   = []
        self.direction_share_weights = []
        self.nonlinear_weights       = []
        self.gaussian_noise = 0.01

        for _ in range(self.n_layers):
            self.share_up_weights.append(initializer.initialize_multiresidual(self.share_up_dim, self.share_up_dim))
            self.share_down_weights.append(initializer.initialize_multiresidual(self.share_down_dim, self.share_down_dim))
            self.softmax_weights.append(initializer.initialize_multiresidual(self.softmax_dim, lambda dims: self.softmax_dim*(2**sum(dims[1:])-1)))
            self.cummax_weights.append(initializer.initialize_multiresidual(self.cummax_dim, self.cummax_dim))
            self.shift_weights.append(initializer.initialize_multiresidual(self.shift_dim, self.shift_dim))
            self.direction_share_weights.append(initializer.initialize_multidirection_share())
            self.nonlinear_weights.append(initializer.initialize_multiresidual(self.nonlinear_dim, self.nonlinear_dim))

        self.head_weights = initializer.initialize_head()
        self.mask_weights = initializer.initialize_linear([1,0,0,1,0], [self.channel_dim_fn([1,0,0,1,0]), 2])

        # Symmetrize all multi-tensor weights
        for lst in [self.share_up_weights, self.share_down_weights,
                    self.softmax_weights, self.cummax_weights,
                    self.shift_weights, self.nonlinear_weights]:
            for w in lst:
                initializer.symmetrize_xy(w)
        for ds in self.direction_share_weights:
            initializer.symmetrize_direction_sharing(ds)

        self.weights_list = initializer.weights_list
    def save_invariant(module, filepath):
        """
        Save all invariant submodules of ARCCompressor to a file.
        Includes spatial_attn, global_attn, temporal_attn, SE, heads, masks, GRU, cross-attn.
        """
        state = {
            # spatial + global attention
            "spatial_attn": module.spatial_attn.state_dict(),
            "global_attn":  module.global_attn.state_dict(),
            # temporal attention
            "temporal_attn": module.temporal_attn.state_dict(),
            # squeeze-and-excitation
            "se_fc1":       module.se_fc1.state_dict(),
            "se_fc2":       module.se_fc2.state_dict(),
            # final heads (plain Tensors)
            "head_w":       module.head_weights[0].detach().cpu(),
            "head_b":       module.head_weights[1].detach().cpu(),
            "mask_w":       module.mask_weights[0].detach().cpu(),
            "mask_b":       module.mask_weights[1].detach().cpu(),
            # GRU and cross-attention
            "gru":          module.gru.state_dict(),
            "ca_q":         module.ca_q.state_dict(),
            "ca_k":         module.ca_k.state_dict(),
            "ca_v":         module.ca_v.state_dict(),
        }
        # object-slot graph-attention
        state.update({
            "obj_q":   module.obj_q.state_dict(),
            "obj_k":   module.obj_k.state_dict(),
            "obj_v":   module.obj_v.state_dict(),
            "obj_out": module.obj_out.state_dict(),
        })
        torch.save(state, filepath)

    # @staticmethod
    def load_invariant(module, filepath, device='cuda'):
        """
        Load all invariant submodules of ARCCompressor from a file.
        """
        try:
            state = torch.load(filepath, map_location=device, weights_only=False)
            # spatial + global attention
            module.spatial_attn.load_state_dict(state["spatial_attn"])
            module.global_attn.load_state_dict(state["global_attn"])
            # temporal attention
            module.temporal_attn.load_state_dict(state["temporal_attn"])
            # squeeze-and-excitation
            module.se_fc1.load_state_dict(state["se_fc1"])
            module.se_fc2.load_state_dict(state["se_fc2"])
            # heads & masks: copy into raw weight Tensors
            hw = state["head_w"].to(device)
            hb = state["head_b"].to(device)
            module.head_weights[0].data.copy_(hw)
            module.head_weights[1].data.copy_(hb)
            mw = state["mask_w"].to(device)
            mb = state["mask_b"].to(device)
            module.mask_weights[0].data.copy_(mw)
            module.mask_weights[1].data.copy_(mb)
            # object-slot graph-attention
            module.obj_q.load_state_dict(state["obj_q"])
            module.obj_k.load_state_dict(state["obj_k"])
            module.obj_v.load_state_dict(state["obj_v"])
            module.obj_out.load_state_dict(state["obj_out"])
            # GRU and cross-attention
            module.gru.load_state_dict(state["gru"])
            module.ca_q.load_state_dict(state["ca_q"])
            module.ca_k.load_state_dict(state["ca_k"])
            module.ca_v.load_state_dict(state["ca_v"])
        except:
            print("Failed to load weights for task ", filepath)
    def forward(self):
        # On-the-fly color permutation during training
        if True:
             perm = torch.arange(self.n_colors_plus1, device=self.problem_orig.device)
             perm[1:] = perm[1:][torch.randperm(self.n_colors_plus1-1)]
 
             # only permute the *training* examples, leave test examples' inputs intact
             # self.problem_orig shape: [n_examples, X, Y, 2]
             permuted = self.problem_orig.clone()
             permuted[:self.task.n_train] = perm[self.problem_orig[:self.task.n_train]]
             self.task.problem = permuted
        else:
            # restore original
            self.task.problem = self.problem_orig

        # Standard decoding pipeline:
        x, KL_amounts, KL_names = layers.decode_latents(
            self.target_capacities, self.decode_weights, self.multiposteriors
        )
        for layer_num in range(self.n_layers):
            x = layers.share_up(x, self.share_up_weights[layer_num]); 
            x = add_noise_multitensor(x, self.gaussian_noise)
            x = layers.softmax(x, self.softmax_weights[layer_num], pre_norm=True, post_norm=False, use_bias=False)
            x = add_noise_multitensor(x, self.gaussian_noise)
            x = layers.cummax(x, self.cummax_weights[layer_num], self.multitensor_system.task.masks, pre_norm=False, post_norm=True, use_bias=False) 
            x = add_noise_multitensor(x, self.gaussian_noise)
            x = layers.shift(x, self.shift_weights[layer_num], self.multitensor_system.task.masks, pre_norm=False, post_norm=True, use_bias=False)
            x = add_noise_multitensor(x, self.gaussian_noise)
            x = layers.direction_share(x, self.direction_share_weights[layer_num], pre_norm=True, use_bias=False)
            x = add_noise_multitensor(x, self.gaussian_noise)
            x = layers.nonlinear(x, self.nonlinear_weights[layer_num], pre_norm=True, post_norm=False, use_bias=False)
            x = add_noise_multitensor(x, self.gaussian_noise)
            x = layers.share_down(x, self.share_down_weights[layer_num])
            x = add_noise_multitensor(x, self.gaussian_noise)
            x = layers.normalize(x)

            # Squeeze-and-Excitation
            core = x[[1,1,0,1,1]]  # [B,C,H,W]
            w = core.mean(dim=(2,3))
            w = torch.relu(self.se_fc1(w)); w = torch.sigmoid(self.se_fc2(w))
            core = core * w[:,:,None,None]
            x[[1,1,0,1,1]] = core

            # Spatial self-attention
            core5d = x[[1,1,0,1,1]]  # [B,Cc,H,W,E]
            B,Cc,H,W,E = core5d.shape
            flat = core5d.view(B*Cc, H*W, E)
            attn_out,_ = self.spatial_attn(flat, flat, flat)
            attn_out = attn_out.view(B,Cc,H,W,E)
            x[[1,1,0,1,1]] = core5d + attn_out

        # — global self‑attention —
        core5d = x[[1, 1, 0, 1, 1]]          # [B, Cc, H, W, E]
        B, Cc, H, W, E = core5d.shape

        # use one token per spatial position (average over the colour axis)
        core_avg = core5d.mean(dim=1)        # [B, H, W, E]

        # flatten spatial grid → sequence
        seq = core_avg.reshape(B, H * W, E)  # [B, L, E]  with L = H·W

        attn_out, _ = self.global_attn(seq, seq, seq)  # MHSA, embed dim = E

        # reshape back to grid
        attn_out = attn_out.view(B, H, W, E)

        # broadcast to every colour slice and residual‑add
        attn_out = attn_out.unsqueeze(1).expand(-1, Cc, -1, -1, -1)  # [B, Cc, H, W, E]
        core5d   = core5d + attn_out
        x[[1, 1, 0, 1, 1]] = core5d

        core5d = x[[1,1,0,1,1]]              # [B, Ccolour, H, W, E]
        slots  = core5d.mean(dim=(2,3))      # [B, Ccolour, E]   ←  one slot / colour

        # ❷  One Graph‑Attention step  (fully‑connected graph on the slots)
        Q = self.obj_q(slots)                # [B, C, E]
        K = self.obj_k(slots)
        V = self.obj_v(slots)

        attn_logits = torch.matmul(Q, K.transpose(-2,-1)) / (E ** 0.5)   # [B, C, C]
        attn        = torch.softmax(attn_logits, dim=-1)

        slots_update = torch.matmul(attn, V)           # message passing
        slots        = slots + self.obj_out(slots_update)   # residual

        # ❸  Broadcast the refined slot vectors back to every pixel
        slots_expanded = slots[:, :, None, None, :]     # [B, C, 1, 1, E]
        core5d         = core5d + slots_expanded
        x[[1,1,0,1,1]] = core5d

        core5d = x[[1,1,0,1,1]]      # [B, Cc, H, W, E]
        B, Cc, H, W, E = core5d.shape
        h     = core5d.mean(dim=1)  # [B, H, W, E]
        h     = h.permute(0,3,1,2)  # [B, E, H, W]

        # prepare slots K,V as before
        slots = core5d.mean(dim=(2,3)).detach()  # [B, Cc, E]
        K, V  = self.ca_k(slots), self.ca_v(slots)

        # UNROLL + COLLECT
        T = 6
        h_seq = []
        for _ in range(T):
            # cross‐attention message (unchanged)
            q   = self.ca_q(h).flatten(2).transpose(1,2)      # [B, L=H·W, E]
            att = torch.softmax(q @ K.transpose(-2,-1) / (E**0.5), dim=-1)  # [B, L, Cc]
            msg = (att @ V).transpose(1,2).view(B, E, H, W)              # [B, E, H, W]

            # ConvGRU update
            h = self.gru(h, msg)    # [B, E, H, W]

            h_seq.append(h)         # collect for temporal attn

        # STACK into [B, T, E, H, W]
        h_seq = torch.stack(h_seq, dim=1)

        # PER‑PIXEL TEMPORAL SELF‑ATTENTION
        flat = h_seq.permute(0, 3, 4, 1, 2).reshape(B*H*W, T, E)
        out, _ = self.temporal_attn(flat, flat, flat)
        out = out.reshape(B, H, W, T, E).mean(dim=3)  # pool over time
 
        # FUSE back into the hidden map
        core5d = core5d + out.unsqueeze(1)             # now [B, 1, H, W, E] broadcasts to [B, Cc, H, W, E]
        x[[1,1,0,1,1]] = core5d
        # -----------------------------------------------------------
        # Final heads
        output = (layers.affine(x[[1,1,0,1,1]], self.head_weights, use_bias=False)
                  + 100*self.head_weights[1])
        x_mask = layers.affine(x[[1,0,0,1,0]], self.mask_weights, use_bias=True)
        y_mask = layers.affine(x[[1,0,0,0,1]], self.mask_weights, use_bias=True)
        x_mask, y_mask = layers.postprocess_mask(self.multitensor_system.task, x_mask, y_mask)
        return output, x_mask, y_mask, KL_amounts, KL_names