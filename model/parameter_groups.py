CLASSIFIER_PARAM_PREFIXES = ("mlp_cls.", "ff_to_cls.", "attentive_fuse_cls.")
BIAS_PARAM_PREFIXES = ("ff_key_projection_bias.",)
BIAS_PARAM_NAMES = {"learnable_vectors_bias"}


def get_param_group_name(name):
    if name.startswith(CLASSIFIER_PARAM_PREFIXES):
        return "classifier"
    if name.startswith(BIAS_PARAM_PREFIXES) or name in BIAS_PARAM_NAMES:
        return "bias"
    return "pred"


def group_named_parameters(model):
    groups = {"pred": [], "classifier": [], "bias": []}
    named_params = list(model.named_parameters())
    for name, param in named_params:
        groups[get_param_group_name(name)].append((name, param))

    group_ids = {
        group_name: {id(param) for _, param in items}
        for group_name, items in groups.items()
    }
    assert group_ids["pred"].isdisjoint(group_ids["classifier"])
    assert group_ids["pred"].isdisjoint(group_ids["bias"])
    assert group_ids["classifier"].isdisjoint(group_ids["bias"])

    all_param_ids = {id(param) for _, param in named_params}
    assigned_param_ids = set().union(*group_ids.values())
    assert assigned_param_ids == all_param_ids
    return groups


def get_model_params_grouped(model):
    groups = group_named_parameters(model)
    pred_params = [param for _, param in groups["pred"]]
    classifier_params = [param for _, param in groups["classifier"]]
    bias_params = [param for _, param in groups["bias"]]
    return pred_params, classifier_params, bias_params


def format_param_group_counts(model):
    groups = group_named_parameters(model)
    counts = {}
    for group_name, items in groups.items():
        counts[group_name] = {
            "tensors": len(items),
            "parameters": sum(param.numel() for _, param in items),
            "trainable_parameters": sum(
                param.numel() for _, param in items if param.requires_grad
            ),
        }

    return (
        "Parameter groups | "
        f"pred: {counts['pred']['tensors']} tensors / {counts['pred']['parameters']:,} params "
        f"({counts['pred']['trainable_parameters']:,} trainable) | "
        f"classifier: {counts['classifier']['tensors']} tensors / {counts['classifier']['parameters']:,} params "
        f"({counts['classifier']['trainable_parameters']:,} trainable) | "
        f"bias: {counts['bias']['tensors']} tensors / {counts['bias']['parameters']:,} params "
        f"({counts['bias']['trainable_parameters']:,} trainable)"
    )
