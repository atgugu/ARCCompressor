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
import torch.nn.functional as F

def add_noise_multitensor(x, sigma):
    # wrap a function that takes a real tensor and adds noise
    return multitensor_systems.multify(
        lambda dims, t: t + sigma * torch.randn_like(t)
    )(x)
class ARCCompressor:
    """
    The main model class for the VAE Decoder in our solution to ARC.
    """

    # Define the channel dimensions that all the layers use
    n_layers = 2
    share_up_dim = 32
    share_down_dim = 16
    decoding_dim = 8
    softmax_dim = 8
    cummax_dim = 8
    shift_dim = 8
    nonlinear_dim = 512 


    # This function gives the channel dimension of the residual stream depending on
    # which dimensions are present, for every tensor in the multitensor.
    def channel_dim_fn(self, dims):
        return 16 if dims[2] == 0 else 8

    def __init__(self, task):
        """
        Create a model that is tailored to the given task, and initialize all the weights.
        The weights are symmetrized such that swapping the x and y dimension ordering should
        make the output's dimension ordering also swapped, for the same weights. This may not
        be exactly correct since symmetrizing all operations is difficult.
        Args:
            task (preprocessing.Task): The task which the model is to be made for solving.
        """
        self.multitensor_system = task.multitensor_system

        # Initialize weights
        initializer = initializers.Initializer(self.multitensor_system, self.channel_dim_fn)

        self.multiposteriors = initializer.initialize_multiposterior(self.decoding_dim)
        self.decode_weights = initializer.initialize_multilinear([self.decoding_dim, self.channel_dim_fn])
        initializer.symmetrize_xy(self.decode_weights)
        self.target_capacities = initializer.initialize_multizeros([self.decoding_dim])

        self.share_up_weights = []
        self.share_down_weights = []
        self.softmax_weights = []
        self.cummax_weights = []
        self.shift_weights = []
        self.direction_share_weights = []
        self.nonlinear_weights = []
        self.gaussian_noise = 0.05
        # --- new: spatial self‑attention on the [in/out] residual ---
        # dims = [1,1,0,1,1] → channel_dim_fn gives us the embed size
        embed_dim = self.channel_dim_fn([1,1,0,1,1])
        self.spatial_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=8,
            batch_first=False
        )   

        # --- new: one *global* MHSA over the whole [in/out] core ---
        # we'll attend over every (color, x, y) position as one long sequence,
        # embedding size = channel_dim_fn([1,1,0,1,1])
        global_embed = self.channel_dim_fn([1,1,0,1,1])
        self.global_attn = nn.MultiheadAttention(
            embed_dim=global_embed,
            num_heads=8,
            batch_first=True
        )


        E = self.channel_dim_fn([1,1,0,1,1])           # latent dim of one pixel
        # we choose 4 heads here, but you can tune num_heads
        self.temporal_attn = nn.MultiheadAttention(embed_dim=E, num_heads=4, batch_first=True)

        # 1‑head Graph‑Attention over “object slots”
        self.obj_q   = nn.Linear(E, E, bias=False)
        self.obj_k   = nn.Linear(E, E, bias=False)
        self.obj_v   = nn.Linear(E, E, bias=False)
        self.obj_out = nn.Linear(E, E, bias=False)

        # --- ConvGRU cell (single gate = GRU‑style, 3×3 conv) ----------------
        class _ConvGRU(nn.Module):
            def __init__(self, E):
                super().__init__()
                self.E = E
                self.conv_zr = nn.Conv2d(E * 2, E * 2, 3, padding=1)  # z & r gates
                self.conv_h  = nn.Conv2d(E * 2, E,     3, padding=1)  # candidate ĥ

            def forward(self, h, x):
                # h,x: [B,E,H,W]
                z, r = torch.split(self.conv_zr(torch.cat([h, x], 1)), self.E, 1)
                z, r = torch.sigmoid(z), torch.sigmoid(r)
                h_hat = torch.tanh(self.conv_h(torch.cat([r * h, x], 1)))
                return (1 - z) * h + z * h_hat
        # ---------------------------------------------------------------------

        E = embed_dim  # from earlier
        self.gru = _ConvGRU(E)

        # linear maps for the cross‑attention (Q from h, K/V from slots)
        self.ca_q = nn.Conv2d(E, E, 1, bias=False)     # 1×1 conv = per‑pixel linear
        self.ca_k = nn.Linear(E, E,  bias=False)
        self.ca_v = nn.Linear(E, E,  bias=False)

        C = self.channel_dim_fn([1,1,0,1,1])  # e.g. 16
        self.se_fc1 = nn.Linear(C, C//4)
        self.se_fc2 = nn.Linear(C//4, C)
        for layer_num in range(self.n_layers):
            self.share_up_weights.append(initializer.initialize_multiresidual(self.share_up_dim, self.share_up_dim))
            self.share_down_weights.append(initializer.initialize_multiresidual(self.share_down_dim, self.share_down_dim))
            output_scaling_fn = lambda dims: self.softmax_dim * (2 ** (dims[1] + dims[2] + dims[3] + dims[4]) - 1)
            self.softmax_weights.append(initializer.initialize_multiresidual(self.softmax_dim, output_scaling_fn))
            self.cummax_weights.append(initializer.initialize_multiresidual(self.cummax_dim, self.cummax_dim))
            self.shift_weights.append(initializer.initialize_multiresidual(self.shift_dim, self.shift_dim))
            self.direction_share_weights.append(initializer.initialize_multidirection_share())
            self.nonlinear_weights.append(initializer.initialize_multiresidual(self.nonlinear_dim, self.nonlinear_dim))

        self.head_weights = initializer.initialize_head()
        self.mask_weights = initializer.initialize_linear(
            [1, 0, 0, 1, 0], [self.channel_dim_fn([1, 0, 0, 1, 0]), 2]
        )

        # Symmetrize weights so that their behavior is equivariant to swapping x and y dimension ordering
        for weight_list in [
            self.share_up_weights,
            self.share_down_weights,
            self.softmax_weights,
            self.cummax_weights,
            self.shift_weights,
            self.nonlinear_weights,
        ]:
            for layer_num in range(self.n_layers):
                initializer.symmetrize_xy(weight_list[layer_num])

        for layer_num in range(self.n_layers):
            initializer.symmetrize_direction_sharing(self.direction_share_weights[layer_num])

        self.weights_list = initializer.weights_list
    @staticmethod
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

    @staticmethod
    def load_invariant(module, filepath, device='cuda'):
        """
        Load all invariant submodules of ARCCompressor from a file.
        """
        state = torch.load(filepath, map_location=device)
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


    def forward(self):
        """
        Compute the forward pass of the VAE decoder. Start by using internally stored latents,
        and process from there. Output an [example, color, x, y, channel] tensor for the colors,
        and an [example, x, channel] and [example, y, channel] tensor for the masks.
        Returns:
            Tensor: An [example, color, x, y, channel] tensor, where for every example,
                    input/output (picked by channel dimension), and every pixel (picked
                    by x and y dimensions), we have a vector full of logits for that
                    pixel being each possible color.
            Tensor: An [example, x, channel] tensor, where for every example, input/output
                    (picked by channel dimension), and every x, we assign a score that
                    contributes to the likelihood that that index of the x dimension is not
                    masked out in the prediction.
            Tensor: An [example, y, channel] tensor, used in the same way as above.
            list[Tensor]: A list of tensors indicating the amount of KL contributed by each component
                    tensor in the layers.decode_latents() step.
            list[str]: A list of tensor names that correspond to each tensor in the aforementioned output.
        """
        # Decoding layer
        x, KL_amounts, KL_names = layers.decode_latents(
            self.target_capacities, self.decode_weights, self.multiposteriors
        )

        for layer_num in range(self.n_layers):
            # Multitensor communication layer
            x = layers.share_up(x, self.share_up_weights[layer_num])
            x = add_noise_multitensor(x, self.gaussian_noise)

            # Softmax layer
            x = layers.softmax(x, self.softmax_weights[layer_num], pre_norm=True, post_norm=False, use_bias=False)
            x = add_noise_multitensor(x, self.gaussian_noise)

            # Directional layers
            x = layers.cummax(
                x, self.cummax_weights[layer_num], self.multitensor_system.task.masks,
                pre_norm=False, post_norm=True, use_bias=False
            )
            x = add_noise_multitensor(x, self.gaussian_noise)

            x = layers.shift(
                x, self.shift_weights[layer_num], self.multitensor_system.task.masks,
                pre_norm=False, post_norm=True, use_bias=False
            )
            x = add_noise_multitensor(x, self.gaussian_noise)


            # Directional communication layer
            x = layers.direction_share(x, self.direction_share_weights[layer_num], pre_norm=True, use_bias=False)
            x = add_noise_multitensor(x, self.gaussian_noise)

            # Nonlinear layer
            x = layers.nonlinear(x, self.nonlinear_weights[layer_num], pre_norm=True, post_norm=False, use_bias=False)
            x = add_noise_multitensor(x, self.gaussian_noise)

            # Multitensor communication layer
            x = layers.share_down(x, self.share_down_weights[layer_num])
            x = add_noise_multitensor(x, self.gaussian_noise)

            # Normalization layer
            x = layers.normalize(x)

            # — Squeeze‑and‑Excitation —
            core = x[[1,1,0,1,1]]          # [B, C, H, W]
            w = core.mean(dim=(2,3))       # [B, C]
            w = torch.relu(self.se_fc1(w))
            w = torch.sigmoid(self.se_fc2(w))  # [B, C]
            core = core * w[:,:,None,None]      # scale each channel
            x[[1,1,0,1,1]] = core
            # — end SE —
            
            # — insert self‑attention over the [in/out] residual —
            # extract the real tensor for dims=[1,1,0,1,1]
            core = x[[1,1,0,1,1]]        # shape: [B, C_color, H, W, E]
            B, Cc, H, W, E = core.shape

            # flatten per‑color spatial positions into a sequence of length H*W
            flat = core.view(B*Cc, H*W, E)    # (batch*n_colors, seq_len, embed)

            # run multi‑head attention (we set batch_first=True)
            attn_out, _ = self.spatial_attn(flat, flat, flat)

            # unflatten and residual‑add
            attn_out = attn_out.view(B, Cc, H, W, E)
            core     = core + attn_out

            # write it back into the MultiTensor at dims [1,1,0,1,1]
            x[[1,1,0,1,1]] = core

        # now project with your existing linear heads

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

        # -----------------------------------------------------------
        # ConvGRU + cross‑attention (T = 3 unrolls)
        core5d = x[[1,1,0,1,1]]              # [B,Cc,H,W,E]
        B, Cc, H, W, E = core5d.shape
        h      = core5d.mean(dim=1)          # initialise hidden with colour‑avg  [B,H,W,E]
        h      = h.permute(0,3,1,2).contiguous()   # -> [B,E,H,W]

        # slots from previous step (already refined by the GAT)
        slots  = slots.detach()              # reuse the 'slots' tensor created earlier
        K = self.ca_k(slots)                 # [B,Cc,E]
        V = self.ca_v(slots)

        for _ in range(24):                   # unroll 3 steps
            # --- Cross‑attention message ------------------------------------
            q   = self.ca_q(h)               # [B,E,H,W]
            q   = q.flatten(2).transpose(1,2)            # [B,L,E]  L = H·W
            att = torch.softmax(
                    torch.matmul(q, K.transpose(-2,-1)) / (E**0.5), dim=-1)  # [B,L,Cc]
            msg = torch.matmul(att, V)                              # [B,L,E]
            msg = msg.transpose(1,2).view(B,E,H,W)                  # [B,E,H,W]

            # --- ConvGRU update --------------------------------------------
            h = self.gru(h, msg)

        # broadcast h back into the multitensor core
        core5d = core5d + h.permute(0,2,3,1).unsqueeze(1)   # [B,Cc,H,W,E]
        x[[1,1,0,1,1]] = core5d
        # -----------------------------------------------------------
        # Linear Heads
        output = (
            layers.affine(x[[1, 1, 0, 1, 1]], self.head_weights, use_bias=False)
            + 100 * self.head_weights[1]
        )
        x_mask = layers.affine(x[[1, 0, 0, 1, 0]], self.mask_weights, use_bias=True)
        y_mask = layers.affine(x[[1, 0, 0, 0, 1]], self.mask_weights, use_bias=True)

        # Postprocessing
        x_mask, y_mask = layers.postprocess_mask(self.multitensor_system.task, x_mask, y_mask)

        return output, x_mask, y_mask, KL_amounts, KL_names

