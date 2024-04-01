from pydantic import BaseModel


class Metadata(BaseModel):
    bacterial_resistance_ampicillin: bool
    bacterial_resistance_chloramphenicol: bool
    bacterial_resistance_kanamycin: bool
    bacterial_resistance_other: bool
    bacterial_resistance_spectinomycin: bool
    copy_number_high_copy: bool
    copy_number_low_copy: bool
    copy_number_unknown: bool
    growth_strain_ccdb_survival: bool
    growth_strain_dh10b: bool
    growth_strain_dh5alpha: bool
    growth_strain_neb_stable: bool
    growth_strain_other: bool
    growth_strain_stbl3: bool
    growth_strain_top10: bool
    growth_strain_xl1_blue: bool
    growth_temp_30: bool
    growth_temp_37: bool
    growth_temp_other: bool
    selectable_markers_blasticidin: bool
    selectable_markers_his3: bool
    selectable_markers_hygromycin: bool
    selectable_markers_leu2: bool
    selectable_markers_neomycin: bool
    selectable_markers_other: bool
    selectable_markers_puromycin: bool
    selectable_markers_trp1: bool
    selectable_markers_ura3: bool
    selectable_markers_zeocin: bool
    species_budding_yeast: bool
    species_fly: bool
    species_human: bool
    species_mouse: bool
    species_mustard_weed: bool
    species_nematode: bool
    species_other: bool
    species_rat: bool
    species_synthetic: bool
    species_zebrafish: bool

    def vector(self) -> list[int]:
        return [
            int(self.bacterial_resistance_ampicillin),
            int(self.bacterial_resistance_chloramphenicol),
            int(self.bacterial_resistance_kanamycin),
            int(self.bacterial_resistance_other),
            int(self.bacterial_resistance_spectinomycin),
            int(self.copy_number_high_copy),
            int(self.copy_number_low_copy),
            int(self.copy_number_unknown),
            int(self.growth_strain_ccdb_survival),
            int(self.growth_strain_dh10b),
            int(self.growth_strain_dh5alpha),
            int(self.growth_strain_neb_stable),
            int(self.growth_strain_other),
            int(self.growth_strain_stbl3),
            int(self.growth_strain_top10),
            int(self.growth_strain_xl1_blue),
            int(self.growth_temp_30),
            int(self.growth_temp_37),
            int(self.growth_temp_other),
            int(self.selectable_markers_blasticidin),
            int(self.selectable_markers_his3),
            int(self.selectable_markers_hygromycin),
            int(self.selectable_markers_leu2),
            int(self.selectable_markers_neomycin),
            int(self.selectable_markers_other),
            int(self.selectable_markers_puromycin),
            int(self.selectable_markers_trp1),
            int(self.selectable_markers_ura3),
            int(self.selectable_markers_zeocin),
            int(self.species_budding_yeast),
            int(self.species_fly),
            int(self.species_human),
            int(self.species_mouse),
            int(self.species_mustard_weed),
            int(self.species_nematode),
            int(self.species_other),
            int(self.species_rat),
            int(self.species_synthetic),
            int(self.species_zebrafish),
        ]


class Input(BaseModel):
    sequence: str
    metadata: Metadata


class ScoredLab(BaseModel):
    id: str
    score: float


class Output(BaseModel):
    ranked_labs: list[ScoredLab]
