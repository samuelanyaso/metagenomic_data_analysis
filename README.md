# Metagenomic data analysis with a case study in geolocation classification

Accompanying bash scripts and `R` codes for the Book chapter **Bioinformatics pre-processing of microbiome data with an application to metagenomic forensics** by Samuel Anyaso-Samuel, Archie Sachdeva, Subharup Guha, Somnath Datta.

A standard pipeline is constructed using bioinformatics tools such as `fastQC`, `multiQC`, `trimmomatic`, and `BowTie2`. Taxonomic profiling is performed using three widely used tools - `MetaPhlAn2`, `Kraken2-Braken` and `Kaiju`. Program codes for implementation of these procedures are provided in the bash scripts folder.

Species abundance table is obtained at the termination of the bioinformatics procedure. Downstream analysis of the abundance table is performed by training several standard supervised learning classifiers, and an ensemble classifier. `R` scripts are provided for the implementation and evaluation of the classifiers based on user-defined performance measures.

