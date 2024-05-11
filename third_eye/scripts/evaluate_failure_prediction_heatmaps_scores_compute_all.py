import glob
import os

from natsort import natsorted

from ThirdEye.ase22.config import Config
from evaluate_failure_prediction_heatmaps_scores import evaluate_failure_prediction

if __name__ == '__main__':
    os.chdir(os.getcwd().replace('scripts', ''))

    cfg = Config()
    cfg.from_pyfile(filename="/mnt/c/Unet/ThirdEye/ase22/config_my.py")

    for condition in ['icse20', 'mutants']:
        simulations = natsorted(glob.glob('/mnt/c/Users/Linfe/Downloads/data-ASE2022/benchmark' + condition + '/*'))

        for ht in ['smoothgrad']:

            for st in ['-avg', '-avg-grad']:

                for am in ['mean', 'max']:

                    for sim in simulations:

                        if "nominal" not in sim:
                            sim = sim.replace("simulations/", "")

                            if "nominal" not in sim or "Normal" not in sim:

                                evaluate_failure_prediction(cfg,
                                                            heatmap_type=ht,
                                                            simulation_name=sim,
                                                            summary_type=st,
                                                            aggregation_method=am,
                                                            condition=condition)
