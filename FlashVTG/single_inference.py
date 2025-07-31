from torch.utils.data import DataLoader
from FlashVTG.start_end_dataset import StartEndDataset, start_end_collate


def get_single_loader(single_json_obj, opt):
    """
    Args:
        single_json_obj (dict): 한 개의 query-video pair 정보. 예: {
            "qid": 2579,
            "query": "A girl and her mother cooked...",
            "duration": 150,
            "vid": "NUsG9BgSes0_210.0_360.0",
            "relevant_clip_ids": [...],
            ...
        }
        opt (Namespace or dict): FlashVTG inference 설정.

    Returns:
        DataLoader: batch_size = 1로 구성된 단일 데이터 inference용 DataLoader
    """
    dataset = StartEndDataset(
        dset_name=opt.dset_name,
        data_path="dummy.jsonl",  # 사용되지 않음
        v_feat_dirs=opt.v_feat_dirs,
        q_feat_dir=opt.t_feat_dir,
        q_feat_type=opt.q_feat_type,
        max_q_l=opt.max_q_l,
        max_v_l=opt.max_v_l,
        ctx_mode=opt.ctx_mode,
        data_ratio=1.0,
        normalize_v=not opt.no_norm_vfeat,
        normalize_t=not opt.no_norm_tfeat,
        clip_len=opt.clip_length,
        max_windows=opt.max_windows,
        load_labels=True,  # True여야 inference target 생성됨
        span_loss_type=opt.span_loss_type,
        txt_drop_ratio=0,
        dset_domain=opt.dset_domain if hasattr(opt, "dset_domain") else None,
    )
    # JSON 1개만 수동 주입
    dataset.data = [single_json_obj]
    dataset._preload_data()

    return DataLoader(
        dataset,
        batch_size=1,
        collate_fn=start_end_collate,
        num_workers=0,
        pin_memory=False,
        shuffle=False,
    )
