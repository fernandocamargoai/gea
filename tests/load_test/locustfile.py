import os

from locust import HttpUser, task
import pandas as pd

from gea.deployment.schema import Input, Metadata


def convert_row_to_input(row: dict) -> Input:
    return Input(
        sequence=row["sequence"],
        metadata=Metadata(
            bacterial_resistance_ampicillin=bool(
                row["bacterial_resistance_ampicillin"]
            ),
            bacterial_resistance_chloramphenicol=bool(
                row["bacterial_resistance_chloramphenicol"]
            ),
            bacterial_resistance_kanamycin=bool(row["bacterial_resistance_kanamycin"]),
            bacterial_resistance_other=bool(row["bacterial_resistance_other"]),
            bacterial_resistance_spectinomycin=bool(
                row["bacterial_resistance_spectinomycin"]
            ),
            copy_number_high_copy=bool(row["copy_number_high_copy"]),
            copy_number_low_copy=bool(row["copy_number_low_copy"]),
            copy_number_unknown=bool(row["copy_number_unknown"]),
            growth_strain_ccdb_survival=bool(row["growth_strain_ccdb_survival"]),
            growth_strain_dh10b=bool(row["growth_strain_dh10b"]),
            growth_strain_dh5alpha=bool(row["growth_strain_dh5alpha"]),
            growth_strain_neb_stable=bool(row["growth_strain_neb_stable"]),
            growth_strain_other=bool(row["growth_strain_other"]),
            growth_strain_stbl3=bool(row["growth_strain_stbl3"]),
            growth_strain_top10=bool(row["growth_strain_top10"]),
            growth_strain_xl1_blue=bool(row["growth_strain_xl1_blue"]),
            growth_temp_30=bool(row["growth_temp_30"]),
            growth_temp_37=bool(row["growth_temp_37"]),
            growth_temp_other=bool(row["growth_temp_other"]),
            selectable_markers_blasticidin=bool(row["selectable_markers_blasticidin"]),
            selectable_markers_his3=bool(row["selectable_markers_his3"]),
            selectable_markers_hygromycin=bool(row["selectable_markers_hygromycin"]),
            selectable_markers_leu2=bool(row["selectable_markers_leu2"]),
            selectable_markers_neomycin=bool(row["selectable_markers_neomycin"]),
            selectable_markers_other=bool(row["selectable_markers_other"]),
            selectable_markers_puromycin=bool(row["selectable_markers_puromycin"]),
            selectable_markers_trp1=bool(row["selectable_markers_trp1"]),
            selectable_markers_ura3=bool(row["selectable_markers_ura3"]),
            selectable_markers_zeocin=bool(row["selectable_markers_zeocin"]),
            species_budding_yeast=bool(row["species_budding_yeast"]),
            species_fly=bool(row["species_fly"]),
            species_human=bool(row["species_human"]),
            species_mouse=bool(row["species_mouse"]),
            species_mustard_weed=bool(row["species_mustard_weed"]),
            species_nematode=bool(row["species_nematode"]),
            species_other=bool(row["species_other"]),
            species_rat=bool(row["species_rat"]),
            species_synthetic=bool(row["species_synthetic"]),
            species_zebrafish=bool(row["species_zebrafish"]),
        ),
    )


class GeneticEngineeringAttributionUser(HttpUser):
    dataset_file = os.environ.get(
        "LOCUST_DATASET_FILE", "artifacts/few_shot_dataset.csv"
    )

    def on_start(self):
        self.input_generator = self._cycle_file(self.dataset_file)

    def _cycle_file(self, filename: str):
        while True:
            df = pd.read_csv(filename)
            for _, row in df.iterrows():
                yield convert_row_to_input(row)

    @task
    def invoke_prediction(self):
        input_ = next(self.input_generator)

        self.client.post(
            f"{self.host}/predict",
            json=input_.dict(),
        )
