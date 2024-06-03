import ThirdEye.ase22.scripts.evaluate_failure_prediction_heatmaps_scores as eva
from ThirdEye.ase22.utils import *

# (cfg, heatmap_type, simulation_name, summary_type, aggregation_method, condition)

if __name__ == '__main__':

    cfg = Config()
    # cfg.from_pyfile(filename="/mnt/c/Unet/ThirdEye/ase22/config_my.py")  # ""
    cfg.from_pyfile(filename="C:/Unet/ThirdEye/ase22/config_my.py")

    # for normal
    segmentation = True
    for ht in ['smoothgrad']:  # for normal remove faster-scorecam, 'faster-scorecam'
        for condition in ['icse20', 'mutants', 'ood']:  # ,
            condition_path = os.path.join(cfg.TESTING_DATA_DIR, condition)
            condition_files = os.listdir(condition_path)
            for simulation_name in condition_files:
                for st in ['-avg-grad', '-avg_road_attention_percentage',
                           '-total_road_attention_percentage']:  #'-avg',
                    for am in ['mean', 'max']:
                        print(condition + '/' + simulation_name)
                        eva.evaluate_failure_prediction(cfg=cfg, heatmap_type=ht,
                                                        simulation_name=condition + '/' + simulation_name,
                                                        summary_type=st, aggregation_method=am, condition=condition,
                                                        segmentation=segmentation)  # mutants
