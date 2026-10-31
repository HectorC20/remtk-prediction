from importlib.machinery import SourceFileLoader
eval_m = SourceFileLoader('eval_m', r'C:\Users\LENOVO\Desktop\ULTIMA DE 2026\remtk-prediction\scripts\evaluate_3_8b_400tools_trained.py').load_module()

cfg_38 = dict(eval_m.CFG)
cfg_39 = dict(eval_m.CFG)
cfg_39['d_exp_ffn'] = 4928

def count_p(cfg):
    m = eval_m.RemtkJuicioJEVMoE3_8B(**cfg)
    total = sum(p.numel() for p in m.parameters())
    moe = sum(p.numel() for p in m.capa2_contexto_moe.moe_bank.parameters())
    set_t = sum(p.numel() for p in m.capa2_contexto_moe.set_layers.parameters())
    return total, moe, set_t

tot_38, moe_38, set_38 = count_p(cfg_38)
tot_39, moe_39, set_39 = count_p(cfg_39)
print(f"CFG 3.8B: Total = {tot_38/1e9:.3f}B (MoE = {moe_38/1e9:.3f}B, Set = {set_38/1e9:.3f}B)")
print(f"CFG 3.9B: Total = {tot_39/1e9:.3f}B (MoE = {moe_39/1e9:.3f}B, Set = {set_39/1e9:.3f}B)")
