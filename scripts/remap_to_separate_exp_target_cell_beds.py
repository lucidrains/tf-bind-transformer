import polars as pl
from pathlib import Path
from tf_bind_transformer.data import read_bed, save_bed

def generate_separate_exp_target_cell_beds(
    remap_file,
    *,
    output_folder = './negative-peaks-per-target',
    exp_target_cell_type_col = 'column_4'
):
    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok = True, parents = True)

    df = read_bed(remap_file)
    target_experiments = df.get_column(exp_target_cell_type_col).unique().to_list()

    for target_experiment in target_experiments:
        filtered_df = df.filter(pl.col(exp_target_cell_type_col) == target_experiment)

        target_bed_path = str(output_folder / f'{target_experiment}.bed')
        save_bed(filtered_df, target_bed_path)

    print('success')
