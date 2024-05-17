import ThirdEye.ase22.scripts.evaluate_failure_prediction_heatmaps_scores as eva
from ThirdEye.ase22.utils import *

# (cfg, heatmap_type, simulation_name, summary_type, aggregation_method, condition)

if __name__ == '__main__':
    heatmap_type = 'smoothgrad'
    cfg = Config()
    cfg.from_pyfile(filename="C:/Unet/ThirdEye/ase22/config_my.py")  # "/mnt/c/Unet/ThirdEye/ase22/config_my.py"
    # cfg.from_pyfile(filename="C:/Unet/ThirdEye/ase22/config_my.py")
    # simulation_name = 'mutants/udacity_add_weights_regularisation_mutated0_MP_l1_3_1'
    simulation_name = 'mutants/udacity_add_weights_regularisation_mutated0_MP_l1_3_1'
        #"xai-track1-fog-20"
    # for st in ['-avg', '-avg_withSeg']

    for st in ['-avg']:
        for am in ['mean', 'max']:
            eva.evaluate_failure_prediction(cfg, heatmap_type, simulation_name, st, am, 'mutants')  # mutants
