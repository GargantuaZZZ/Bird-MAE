import hydra
from omegaconf import DictConfig
from models import AudioMAE, EAT, VIT_EAT, VIT, VIT_SourceSeparation, VIT_Contrastive, ConvNext, VIT_ppnet, VIT_ppnet_SourceSeparation, VIT_MIM, BirdAVES, BirdAVES_ppnet, SimCLR, SimCLR_ppnet
from models.jepa.models_jepa import A_JEPA, VIT_JEPA
from util import pylogger

log = pylogger.get_pylogger(__name__)

def instantiate_callbacks(cfg_callbacks: DictConfig):
    callbacks = []

    if not cfg_callbacks:
        log.warning("No callbacks found")

    for _, cb_config in cfg_callbacks.items():
        if isinstance(cb_config, DictConfig) and "_target_" in cb_config:
            log.info(f"Instantiating callback <{cb_config._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_config))
    return callbacks


def build_model(cfg_module: DictConfig):
    if cfg_module.network.name == "AudioMAE":
        module = AudioMAE(
            norm_layer=cfg_module.network.norm_layer,
            norm_pix_loss=cfg_module.network.norm_pix_loss,
            mask_ratio=cfg_module.network.mask_ratio,
            cfg_encoder=cfg_module.network.encoder,
            cfg_decoder=cfg_module.network.decoder,
            optimizer=cfg_module.optimizer,
            scheduler=cfg_module.scheduler,
            loss=cfg_module.loss,
            
        )
    elif cfg_module.network.name == "VIT":
        module = VIT(
            img_size_x=cfg_module.network.img_size_x,
            img_size_y=cfg_module.network.img_size_y,
            patch_size=cfg_module.network.patch_size,
            in_chans=cfg_module.network.in_chans,
            embed_dim=cfg_module.network.embed_dim,
            global_pool=cfg_module.network.global_pool,
            drop_path=cfg_module.network.drop_path,
            norm_layer=cfg_module.network.norm_layer,
            mlp_ratio=cfg_module.network.mlp_ratio,
            qkv_bias=cfg_module.network.qkv_bias,
            eps=cfg_module.network.eps,
            num_heads=cfg_module.network.num_heads,
            depth=cfg_module.network.depth,
            pretrained_weights_path=cfg_module.network.pretrained_weights_path,
            target_length=cfg_module.network.target_length,
            num_classes=cfg_module.network.num_classes,
            optimizer=cfg_module.optimizer,
            scheduler=cfg_module.scheduler,
            loss=cfg_module.loss,
            metric_cfg=cfg_module.metric,
            mask2d=cfg_module.network.mask2d,
            mask_t_prob=cfg_module.network.mask_t_prob,
            mask_f_prob=cfg_module.network.mask_f_prob,
            ema_update_rate=cfg_module.network.ema_update_rate,
            mask_inference=cfg_module.network.mask_inference,
        )
    elif cfg_module.network.name == "VIT_Contrastive":
        contrastive_cfg = cfg_module.network.get("contrastive", {})
        module = VIT_Contrastive(
            img_size_x=cfg_module.network.img_size_x,
            img_size_y=cfg_module.network.img_size_y,
            patch_size=cfg_module.network.patch_size,
            in_chans=cfg_module.network.in_chans,
            embed_dim=cfg_module.network.embed_dim,
            global_pool=cfg_module.network.global_pool,
            drop_path=cfg_module.network.drop_path,
            norm_layer=cfg_module.network.norm_layer,
            mlp_ratio=cfg_module.network.mlp_ratio,
            qkv_bias=cfg_module.network.qkv_bias,
            eps=cfg_module.network.eps,
            num_heads=cfg_module.network.num_heads,
            depth=cfg_module.network.depth,
            pretrained_weights_path=cfg_module.network.pretrained_weights_path,
            target_length=cfg_module.network.target_length,
            num_classes=cfg_module.network.num_classes,
            optimizer=cfg_module.optimizer,
            scheduler=cfg_module.scheduler,
            loss=cfg_module.loss,
            metric_cfg=cfg_module.metric,
            mask2d=cfg_module.network.mask2d,
            mask_t_prob=cfg_module.network.mask_t_prob,
            mask_f_prob=cfg_module.network.mask_f_prob,
            ema_update_rate=cfg_module.network.ema_update_rate,
            mask_inference=cfg_module.network.mask_inference,
            contrastive_weight=contrastive_cfg.get("weight", 0.1),
            contrastive_temperature=contrastive_cfg.get("temperature", 0.07),
            contrastive_dim=contrastive_cfg.get("out_dim", 128),
            contrastive_hidden_dim=contrastive_cfg.get("hidden_dim", 512),
        )

    elif cfg_module.network.name == "VIT_SourceSeparation":
        module = VIT_SourceSeparation(
            img_size_x=cfg_module.network.img_size_x,
            img_size_y=cfg_module.network.img_size_y,
            patch_size=cfg_module.network.patch_size,
            in_chans=cfg_module.network.in_chans,
            embed_dim=cfg_module.network.embed_dim,
            global_pool=cfg_module.network.global_pool,
            drop_path=cfg_module.network.drop_path,
            norm_layer=cfg_module.network.norm_layer,
            mlp_ratio=cfg_module.network.mlp_ratio,
            qkv_bias=cfg_module.network.qkv_bias,
            eps=cfg_module.network.eps,
            num_heads=cfg_module.network.num_heads,
            depth=cfg_module.network.depth,
            pretrained_weights_path=cfg_module.network.pretrained_weights_path,
            target_length=cfg_module.network.target_length,
            num_classes=cfg_module.network.num_classes,
            optimizer=cfg_module.optimizer,
            scheduler=cfg_module.scheduler,
            loss=cfg_module.loss,
            metric_cfg=cfg_module.metric,
            mask2d=cfg_module.network.mask2d,
            mask_t_prob=cfg_module.network.mask_t_prob,
            mask_f_prob=cfg_module.network.mask_f_prob,
            ema_update_rate=cfg_module.network.ema_update_rate,
            mask_inference=cfg_module.network.mask_inference,
            source_aggregation=cfg_module.network.source_aggregation,
            source_top_k=cfg_module.network.source_top_k,
            source_max_weight=cfg_module.network.source_max_weight,
            source_attention_hidden_dim=cfg_module.network.get("source_attention_hidden_dim", 64),
            source_attention_dropout=cfg_module.network.get("source_attention_dropout", 0.0),
            source_attention_classwise=cfg_module.network.get("source_attention_classwise", True),
            source_gate_init=cfg_module.network.get("source_gate_init", 0.5),
            source_gate_classwise=cfg_module.network.get("source_gate_classwise", True),
        )

    elif cfg_module.network.name == "VIT_ppnet":
        module = VIT_ppnet(
            img_size_x=cfg_module.network.img_size_x,
            img_size_y=cfg_module.network.img_size_y,
            patch_size=cfg_module.network.patch_size,
            in_chans=cfg_module.network.in_chans,
            embed_dim=cfg_module.network.embed_dim,
            global_pool=cfg_module.network.global_pool,
            drop_path=cfg_module.network.drop_path,
            norm_layer=cfg_module.network.norm_layer,
            mlp_ratio=cfg_module.network.mlp_ratio,
            qkv_bias=cfg_module.network.qkv_bias,
            eps=cfg_module.network.eps,
            num_heads=cfg_module.network.num_heads,
            depth=cfg_module.network.depth,
            pretrained_weights_path=cfg_module.network.pretrained_weights_path,
            target_length=cfg_module.network.target_length,
            num_classes=cfg_module.network.num_classes,
            optimizer=cfg_module.optimizer,
            scheduler=cfg_module.scheduler,
            loss=cfg_module.loss,
            metric_cfg=cfg_module.metric,
            mask2d=cfg_module.network.mask2d,
            mask_t_prob=cfg_module.network.mask_t_prob,
            mask_f_prob=cfg_module.network.mask_f_prob,
            ema_update_rate=cfg_module.network.ema_update_rate,
            ppnet_cfg=cfg_module.network.ppnet,
            mask_inference=cfg_module.network.mask_inference
        )

    elif cfg_module.network.name == "VIT_ppnet_SourceSeparation":
        module = VIT_ppnet_SourceSeparation(
            img_size_x=cfg_module.network.img_size_x,
            img_size_y=cfg_module.network.img_size_y,
            patch_size=cfg_module.network.patch_size,
            in_chans=cfg_module.network.in_chans,
            embed_dim=cfg_module.network.embed_dim,
            global_pool=cfg_module.network.global_pool,
            drop_path=cfg_module.network.drop_path,
            norm_layer=cfg_module.network.norm_layer,
            mlp_ratio=cfg_module.network.mlp_ratio,
            qkv_bias=cfg_module.network.qkv_bias,
            eps=cfg_module.network.eps,
            num_heads=cfg_module.network.num_heads,
            depth=cfg_module.network.depth,
            pretrained_weights_path=cfg_module.network.pretrained_weights_path,
            target_length=cfg_module.network.target_length,
            num_classes=cfg_module.network.num_classes,
            optimizer=cfg_module.optimizer,
            scheduler=cfg_module.scheduler,
            loss=cfg_module.loss,
            metric_cfg=cfg_module.metric,
            mask2d=cfg_module.network.mask2d,
            mask_t_prob=cfg_module.network.mask_t_prob,
            mask_f_prob=cfg_module.network.mask_f_prob,
            ema_update_rate=cfg_module.network.ema_update_rate,
            ppnet_cfg=cfg_module.network.ppnet,
            mask_inference=cfg_module.network.mask_inference,
            source_aggregation=cfg_module.network.source_aggregation,
            source_top_k=cfg_module.network.source_top_k,
            source_max_weight=cfg_module.network.source_max_weight,
            source_attention_hidden_dim=cfg_module.network.get("source_attention_hidden_dim", 64),
            source_attention_dropout=cfg_module.network.get("source_attention_dropout", 0.0),
            source_attention_classwise=cfg_module.network.get("source_attention_classwise", True),
            source_gate_init=cfg_module.network.get("source_gate_init", 0.5),
            source_gate_classwise=cfg_module.network.get("source_gate_classwise", True),
        )

    elif cfg_module.network.name == "VIT_MIM":
        module = VIT_MIM(
            img_size_x=cfg_module.network.img_size_x,
            img_size_y=cfg_module.network.img_size_y,
            patch_size=cfg_module.network.patch_size,
            in_chans=cfg_module.network.in_chans,
            embed_dim=cfg_module.network.embed_dim,
            global_pool=cfg_module.network.global_pool,
            drop_path=cfg_module.network.drop_path,
            norm_layer=cfg_module.network.norm_layer,
            mlp_ratio=cfg_module.network.mlp_ratio,
            qkv_bias=cfg_module.network.qkv_bias,
            eps=cfg_module.network.eps,
            num_heads=cfg_module.network.num_heads,
            depth=cfg_module.network.depth,
            pretrained_weights_path=cfg_module.network.pretrained_weights_path,
            target_length=cfg_module.network.target_length,
            mim_cfg=cfg_module.network.mim,
            optimizer_cfg=cfg_module.optimizer
        )
    elif cfg_module.network.name == "ConvNext":
        module = ConvNext(
            num_channels=cfg_module.network.num_channels,
            num_classes=cfg_module.network.num_classes,
            hf_checkpoint=cfg_module.network.hf_checkpoint,
            model_dir=cfg_module.network.model_dir,
            optimizer=cfg_module.optimizer,
            scheduler=cfg_module.scheduler,
            loss=cfg_module.loss,
            metric_cfg=cfg_module.metric  
        )
    elif cfg_module.network.name == "A_JEPA":
        module = A_JEPA(
            cfg_encoder=cfg_module.network.encoder,
            cfg_predictor=cfg_module.network.predictor,
            cfg_optimizer=cfg_module.optimizer,
        )
    elif cfg_module.network.name == "VIT_JEPA":
        module = VIT_JEPA(
            img_size=cfg_module.network.img_size,
            patch_size=cfg_module.network.patch_size,
            in_chans=cfg_module.network.in_chans,
            embed_dim=cfg_module.network.embed_dim,
            mlp_ratio=cfg_module.network.mlp_ratio,
            qkv_bias=cfg_module.network.qkv_bias,
            num_heads=cfg_module.network.num_heads,
            depth=cfg_module.network.depth,
            pretrained_weights_path=cfg_module.network.pretrained_weights_path,
            num_classes=cfg_module.network.num_classes,
            optimizer=cfg_module.optimizer,
            scheduler=cfg_module.scheduler,
            loss=cfg_module.loss,
            metric_cfg=cfg_module.metric,
            drop_rate=cfg_module.network.drop_rate,
            qk_scale=cfg_module.network.qk_scale,
            attn_drop_rate=cfg_module.network.attn_drop_rate,
            drop_path_rate=cfg_module.network.drop_path_rate,
            init_std=cfg_module.network.init_std,
            target_length=cfg_module.network.target_length,
        )
    
    elif cfg_module.network.name == "EAT":
        module = EAT(
            norm_layer=cfg_module.network.norm_layer,
            mask_ratio=cfg_module.network.mask_ratio,
            cfg_encoder=cfg_module.network.encoder,
            cfg_decoder=cfg_module.network.decoder,
            cfg_teacher=cfg_module.network.teacher,
            cfg_teacher_assistant=cfg_module.network.teacher_assistant,
            cfg_task=cfg_module.network.task,
            optimizer=cfg_module.network.optimizer,
            scheduler=cfg_module.scheduler,
            compile_mode=cfg_module.network.compile_mode,
        )
    
    elif cfg_module.network.name == "VIT_EAT":
        module = VIT_EAT(
            img_size_x=cfg_module.network.img_size_x,
            img_size_y=cfg_module.network.img_size_y,
            patch_size=cfg_module.network.patch_size,
            in_chans=cfg_module.network.in_chans,
            embed_dim=cfg_module.network.embed_dim,
            global_pool=cfg_module.network.global_pool,
            norm_layer=cfg_module.network.norm_layer,
            mlp_ratio=cfg_module.network.mlp_ratio,
            qkv_bias=cfg_module.network.qkv_bias,
            eps=cfg_module.network.eps,
            drop_path=cfg_module.network.drop_path,
            num_heads=cfg_module.network.num_heads,
            depth=cfg_module.network.depth,
            num_classes=cfg_module.network.num_classes,
            optimizer=cfg_module.optimizer,
            scheduler=cfg_module.scheduler,
            pretrained_weights_path=cfg_module.network.pretrained_weights_path,
            target_length=cfg_module.network.target_length,
            loss=cfg_module.loss,
            metric_cfg=cfg_module.metric,
            mask_t_prob=cfg_module.network.mask_t_prob,
            mask_f_prob=cfg_module.network.mask_f_prob,
            mask2d=cfg_module.network.mask2d,
            mask_mode=cfg_module.network.get("mask_mode", "rand"),
            pos_trainable=cfg_module.network.get("pos_trainable", False),
            ppnet_cfg=cfg_module.network.ppnet_cfg
        )

    elif cfg_module.network.name == "BirdAVES-large":
        module = BirdAVES(num_classes=cfg_module.network.num_classes,
                          optimizer=cfg_module.optimizer,
                          scheduler=cfg_module.scheduler,
                          loss=cfg_module.loss,
                          metric_cfg=cfg_module.metric,
                          birdaves_cfg_path=cfg_module.network.birdaves_cfg,
                          birdaves_weights_path=cfg_module.network.birdaves_weights_path,
                          )

    elif cfg_module.network.name == "BirdAVES-large-ppnet":
        module = BirdAVES_ppnet(num_classes=cfg_module.network.num_classes,
                          optimizer=cfg_module.optimizer,
                          scheduler=cfg_module.scheduler,
                          loss=cfg_module.loss,
                          metric_cfg=cfg_module.metric,
                          birdaves_cfg_path=cfg_module.network.birdaves_cfg,
                          birdaves_weights_path=cfg_module.network.birdaves_weights_path,
                          ppnet_cfg=cfg_module.network.ppnet,
                          )

    elif cfg_module.network.name == "ProtoCLR-simclr":
        module = SimCLR(num_classes=cfg_module.network.num_classes,
                        optimizer=cfg_module.optimizer,
                        scheduler=cfg_module.scheduler,
                        loss=cfg_module.loss,
                        metric_cfg=cfg_module.metric,
                        model_spec_cfg=cfg_module.network.model_spec_cfg,
                        proto_clr_weights_path=cfg_module.network.proto_clr_weights_path
                        )

    elif cfg_module.network.name == "ProtoCLR-simclr-ppnet":
        module = SimCLR_ppnet(num_classes=cfg_module.network.num_classes,
                        optimizer=cfg_module.optimizer,
                        scheduler=cfg_module.scheduler,
                        loss=cfg_module.loss,
                        metric_cfg=cfg_module.metric,
                        model_spec_cfg=cfg_module.network.model_spec_cfg,
                        proto_clr_weights_path=cfg_module.network.proto_clr_weights_path,
                        ppnet_cfg=cfg_module.network.ppnet
                        )
    else:
        raise ValueError(f"Model {cfg_module.network.name} not found")

    return module
