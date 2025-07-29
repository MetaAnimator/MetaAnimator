import torch
import torch.nn as nn

from cldm.ddim_hacked import DDIMSampler
from cldm.cldm import ControlNet, ControlLDM
from cldm.lora import LoRALinearLayer, LoRACompatibleLinear
from ldm.modules.diffusionmodules.util import timestep_embedding


class ControlNetFinetune(ControlNet):
    def __init__(self, ft_with_lora=False, lora_rank=128, norm_trainable=True, zero_trainable=True, 
                 lora_num=1, train_one_out_of_two=False, vae_fusion=False, train_Temporal_Layers_only=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ft_with_lora = ft_with_lora
        self.lora_rank = lora_rank
        self.norm_trainable = norm_trainable
        self.zero_trainable = zero_trainable
        self.lora_num = lora_num
        self.train_one_out_of_two = train_one_out_of_two
        self.vae_fusion = vae_fusion
        self.train_Temporal_Layers_only = train_Temporal_Layers_only

        # delete input hint block
        del self.input_hint_block

        if ft_with_lora:
            # replace linear with lora linear
            for n, m in self.named_modules():
                if isinstance(m, nn.Linear):
                    # define lora linear
                    lora_layer = LoRALinearLayer(m.in_features, m.out_features, rank=lora_rank)
                    lora_linear = LoRACompatibleLinear(m.in_features, m.out_features, lora_layer=lora_layer)
                    lora_linear.weight.data.copy_(m.weight.data)
                    if hasattr(m, 'bias') and m.bias is not None:
                        lora_linear.bias.data.copy_(m.bias.data)  # type: ignore
                    else:
                        lora_linear.bias = None
                    # replace linear with lora linear
                    parent = self
                    *path, name = n.split('.')
                    while path:
                        parent = parent.get_submodule(path.pop(0))
                    parent._modules[name] = lora_linear

    def forward(self, hint, timesteps, context, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        outs = []

        h = hint.type(self.dtype)
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))

        return outs


class ControlFinetuneLDM_original(ControlLDM):

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        ddim_sampler = DDIMSampler(self)
        b, c, h, w = cond["c_concat"][0].shape
        shape = (self.channels, h // 8, w // 8)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        return samples, intermediates

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        cond_txt = torch.cat(cond['c_crossattn'], 1) # 这一行的作用：如果有多个text prompt，将它们沿着dim=1拼接起来；如果只有一个text prompt，那么就是原样

        if cond['c_concat'] is None:
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
        else:
            hint = torch.cat(cond['c_concat'], 1)
            hint = self.get_first_stage_encoding(self.encode_first_stage(hint))
            control = self.control_model(hint=hint, timesteps=t, context=cond_txt)
            control = [c * scale for c, scale in zip(control, self.control_scales)]
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)

        return eps

    def configure_optimizers(self):
        lr = self.learning_rate
        params = []
        f = open('./tmp/finetune_trainable_params.txt', 'w')
        for n, p in self.control_model.named_parameters():
            assert 'input_hint' not in n
            if self.control_model.ft_with_lora:  # only train lora layers, zero convs and norm layers
                if 'lora_layer' in n:  # lora layers
                    params.append(p)
                    f.write(n + '\n')
                elif ('zero_convs' in n or 'middle_block_out' in n) and self.control_model.zero_trainable:  # zero convs
                    # note that middle_block_out is also a zero conv added by controlnet!
                    params.append(p)
                    f.write(n + '\n')
                elif 'norm' in n and self.control_model.norm_trainable:  # norm layers
                    params.append(p)
                    f.write(n + '\n')
            else:
                assert 'lora_layer' not in n
                params.append(p)
                f.write(n + '\n')
        opt = torch.optim.AdamW(params, lr=lr)
        print(f'Optimizable params: {sum([p.numel() for p in params])/1e6:.1f}M')
        f.close()
        return opt


class SimpleMLP(nn.Module):
    def __init__(self, input_channels=8, output_channels=4, hidden_channels=16):
        super(SimpleMLP, self).__init__()
        self.layer1 = nn.Linear(input_channels, hidden_channels)
        self.layer2 = nn.Linear(hidden_channels, output_channels)
        self.relu = nn.ReLU()
    def forward(self, x):
        # Concatenate the input along the channel dimension（输入是hint_list，长度为2，包含两个hint，所以刚开始需要将两个[1, 4, 64, 64]concate为[1, 8, 64, 64]）
        x = torch.cat(x, 1) 
        # Flatten the input except the batch dimension
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, channels, -1).permute(0, 2, 1)
        # Apply the MLP layers
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        # Reshape back to the original spatial dimensions
        x = x.permute(0, 2, 1).view(batch_size, -1, height, width)
        return x


class AttentionConvFusionModule(nn.Module):
    def __init__(self, in_channels, out_channels, attention_heads=8):
        """
        结合卷积和注意力机制的特征融合模块，保持输出维度不变。
        参数：
        - in_channels: 输入特征图的通道数
        - out_channels: 输出特征图的通道数
        - attention_heads: 自注意力的头数
        """
        super(AttentionConvFusionModule, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # 1x1 卷积层：用于保持输入和输出特征图的通道数一致
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(out_channels*2, out_channels, kernel_size=1)
        # Zero Conv层：初始化为零的卷积层
        self.zero_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        nn.init.constant_(self.zero_conv.weight, 0)
        nn.init.constant_(self.zero_conv.bias, 0)
        # Self-attention机制
        self.attention = nn.MultiheadAttention(embed_dim=out_channels*2, num_heads=attention_heads, batch_first=True)
        # 激活函数
        self.relu = nn.ReLU()
        # 自适应权重
        self.alpha = nn.Parameter(torch.tensor(0.0))
    def forward(self, x):
        feature_1, feature_2 = x
        # 1. 对两个特征图进行卷积，保持通道数一致
        feature_1 = self.conv1(feature_1)
        feature_2 = self.conv2(feature_2)
        # ReLU 激活
        feature_1 = self.relu(feature_1)
        feature_2 = self.relu(feature_2)
        # 2. 融合两个特征图，按通道拼接
        fused_features = torch.cat((feature_1, feature_2), dim=1)  # 拼接后通道数为 2 * out_channels
        # 3. 将融合后的特征进行自注意力机制处理
        batch_size, channels, height, width = fused_features.shape
        # 将特征图从 (batch_size, channels, height, width) 转换为 (batch_size, height * width, channels)
        fused_features_flat = fused_features.view(batch_size, channels, -1).permute(0, 2, 1)
        # 使用注意力机制进行特征调整
        attention_output, _ = self.attention(fused_features_flat, fused_features_flat, fused_features_flat)
        # 4. 恢复特征形状回到 (batch_size, channels, height, width)
        attention_output = attention_output.permute(0, 2, 1).view(batch_size, channels, height, width)
        # 5. 最后的卷积操作，确保输出维度不变
        fusion_output = self.conv3(attention_output)
        zero_output = self.zero_conv(fusion_output)
        
        # output = zero_output + feature_1
        output = self.alpha * zero_output + (1 - self.alpha) * feature_1
        return output


class ControlFinetuneLDM(ControlLDM):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lora_weights = [1.0 / self.control_model.lora_num] * self.control_model.lora_num # [0.5, 0.5]
        if self.control_model.vae_fusion:
            # self.fusion_module = SimpleMLP(input_channels=8, output_channels=4, hidden_channels=16)
            self.fusion_module = AttentionConvFusionModule(in_channels=4, out_channels=4)

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        ddim_sampler = DDIMSampler(self)
        b, c, h, w = cond["c_concat"][0].shape
        shape = (self.channels, h // 8, w // 8)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        return samples, intermediates

    def apply_model(self, x_noisy, t, conds, *args, **kwargs):
        if self.control_model.lora_num == 1:
            if self.control_model.vae_fusion:
                if isinstance(conds, dict):
                    conds = [conds]
                diffusion_model = self.model.diffusion_model
                cond_txt = torch.cat(conds[0]['c_crossattn'], 1) # 这一行的作用：如果有多个text prompt，将它们沿着dim=1拼接起来；如果只有一个text prompt，那么就是原样
                if conds[0]['c_concat'] is None or conds[1]['c_concat'] is None:
                    eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
                    return eps
                hint_list = []
                for i, cond in enumerate(conds):
                    hint = torch.cat(cond['c_concat'], 1)
                    hint = self.get_first_stage_encoding(self.encode_first_stage(hint))
                    hint_list.append(hint)
                # =============这里的fusion方式还得确认一下
                # hint_fusion = torch.cat(hint_list, 1)
                hint_fusion = self.fusion_module(hint_list)
                # hint_fusion = hint_list[0] + hint_list[1] # torch.Size([1, 4, 64, 64])
                # =============
                control = self.control_model(hint=hint_fusion, timesteps=t, context=cond_txt)
                control = [c * scale for c, scale in zip(control, self.control_scales)]
                eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)
                del hint_list
            else:
                assert isinstance(conds, dict)
                diffusion_model = self.model.diffusion_model
                cond_txt = torch.cat(conds['c_crossattn'], 1) # 这一行的作用：如果有多个text prompt，将它们沿着dim=1拼接起来；如果只有一个text prompt，那么就是原样
                if conds['c_concat'] is None:
                    eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
                else:
                    hint = torch.cat(conds['c_concat'], 1)
                    hint = self.get_first_stage_encoding(self.encode_first_stage(hint))
                    control = self.control_model(hint=hint, timesteps=t, context=cond_txt)
                    control = [c * scale for c, scale in zip(control, self.control_scales)]
                    eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)                    
               
        elif self.control_model.lora_num == 2:
            if isinstance(conds, dict):
                conds = [conds]
            assert isinstance(conds, (list, tuple))
            assert len(conds) == self.control_model.lora_num
            assert len(self.lora_weights) == self.control_model.lora_num
            weights = self.lora_weights
            diffusion_model = self.model.diffusion_model
            cond_txt = torch.cat(conds[0]['c_crossattn'], 1) # 这一行的作用：如果有多个text prompt，将它们沿着dim=1拼接起来；如果只有一个text prompt，那么就是原样
            
            if conds[0]['c_concat'] is None or conds[1]['c_concat'] is None:
                eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
                return eps
            else:
                controls = []
                for i, cond in enumerate(conds):
                    hint = torch.cat(cond['c_concat'], 1)
                    hint = self.get_first_stage_encoding(self.encode_first_stage(hint))
                    if i == 0:
                        control = self.control_model(hint=hint, timesteps=t, context=cond_txt)
                    else:
                        control = self.control_model_add(hint=hint, timesteps=t, context=cond_txt, weights=weights[i])
                    control = [c * scale for c, scale in zip(control, self.control_scales)]
                    controls.append(control)
                control = [c * weights[0] for c in controls[0]]  # 第一个controlnet的输出
                for i in range(1, len(controls)):  # 将后面的controlnet的输出加到第一个controlnet的输出上
                    control = [c + controls[i][j] * weights[i] for j, c in enumerate(control)]
                eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)
        return eps
            

    def configure_optimizers(self):
        lr = self.learning_rate
        params = []
        f = open('./tmp/finetune_trainable_params.txt', 'w')
        # ========================= 将Motion Module的参数加入到params中
        if self.control_model.train_Temporal_Layers_only:
            print('========================= Only train Temporal Layers!!! =========================')
            if hasattr(self.model.diffusion_model, 'input_blocks_motion_module'):
                for n, p in self.model.diffusion_model.input_blocks_motion_module.named_parameters():
                    params.append(p)
                    f.write(n + '\n')
            if hasattr(self.model.diffusion_model, 'output_blocks_motion_module'):
                for n, p in self.model.diffusion_model.output_blocks_motion_module.named_parameters():
                    params.append(p)
                    f.write(n + '\n')
        # =========================
        else:
            # ========================= 将self.fusion_module的参数加入到params中
            if hasattr(self, 'fusion_module'):
                for n, p in self.fusion_module.named_parameters():
                    params.append(p)
                    f.write(n + '\n')
            # =========================
            for n, p in self.control_model.named_parameters():
                assert 'input_hint' not in n
                if self.control_model.ft_with_lora:  # only train lora layers, zero convs and norm layers
                    # =================================================== 如果只训练两个LoRA中的第二个LoRA，那么就跳过第一个LoRA的参数
                    if self.control_model.train_one_out_of_two:
                        continue
                    # ===================================================
                    else:
                        if 'lora_layer' in n:  # lora layers
                            params.append(p)
                            f.write(n + '\n')
                        elif ('zero_convs' in n or 'middle_block_out' in n) and self.control_model.zero_trainable:  # zero convs
                            # note that middle_block_out is also a zero conv added by controlnet!
                            params.append(p)
                            f.write(n + '\n')
                        elif 'norm' in n and self.control_model.norm_trainable:  # norm layers
                            params.append(p)
                            f.write(n + '\n')
                else:
                    assert 'lora_layer' not in n
                    params.append(p)
                    f.write(n + '\n')
            if self.control_model.lora_num == 2:
                for n, p in self.control_model_add.named_parameters():
                    assert 'input_hint' not in n
                    if self.control_model_add.ft_with_lora:  # only train lora layers, zero convs and norm layers
                        if 'lora_layer' in n:  # lora layers
                            params.append(p)
                            f.write(n + '\n')
                        elif ('zero_convs' in n or 'middle_block_out' in n) and self.control_model_add.zero_trainable:  # zero convs
                            # note that middle_block_out is also a zero conv added by controlnet!
                            params.append(p)
                            f.write(n + '\n')
                        elif 'norm' in n and self.control_model_add.norm_trainable:  # norm layers
                            params.append(p)
                            f.write(n + '\n')
                    else:
                        assert 'lora_layer' not in n
                        params.append(p)
                        f.write(n + '\n')
        opt = torch.optim.AdamW(params, lr=lr)
        print(f'Optimizable params: {sum([p.numel() for p in params])/1e6:.1f}M')
        f.close()
        return opt

    
    