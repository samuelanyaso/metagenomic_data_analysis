#!/bin/sh
#SBATCH --job-name=preQC				# Job name
#SBATCH --account=group_name			# your own sponser or account
#SBATCH --qos=group_name                # your own sponser or account
#SBATCH --mail-type=ALL               	# Mail events
#SBATCH --mail-user=name@email.com      # Where to send email	
#SBATCH --ntasks=1                    	# Run on a single machine (node)
#SBATCH --cpus-per-task 30		  		# Run on a several CPU
#SBATCH --mem-per-cpu=500mb             # Memory limit
#SBATCH --time=01:00:00               	# Time: hrs:min:sec
#SBATCH --output=output_%j.out  	 	# Output and error log 

pwd; hostname; date 

# ############################################################
# Assess quality using fastQC and multiQC
# ############################################################

cd /path/to/pair_end/fastq/files
mkdir fastqc_output

# Loads the fastQC program
module load fastqc/0.11.7

# Peforms quality checks on all .fastqc files using FastQC
fastqc -t 30 *.fastq -o fastqc_output/

# Aggregate the results of fastqc quality control checks using MultiQC
cd /path/to/fastqc_output

# Loads the multiQC program
module load multiqc/1.7

# Runs the multiqc program
multiqc *_fastqc.zip

echo "pre-QC analysis complete"

date


