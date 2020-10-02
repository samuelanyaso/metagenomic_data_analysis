#!/bin/sh
#SBATCH --job-name=postQC				# Job name
#SBATCH --account=group_name			# your own sponser or account
#SBATCH --qos=group_name                # your own sponser or account
#SBATCH --mail-type=ALL               	# Mail events
#SBATCH --mail-user=name@email.com      # Where to send email	
#SBATCH --ntasks=1                    	# Run on a single machine (node)
#SBATCH --cpus-per-task 30		  		# Run on a several CPU
#SBATCH --mem-per-cpu=500mb             # Memory limit
#SBATCH --time=01:00:00               	# Time: hrs:min:sec
#SBATCH --output=serial_test_%j.out   	# Output and error log 

pwd; hostname; date 


# ############################################################
# Perform post-QC analysis using fastQC and multiQC
# ############################################################

# We begin post QC checks
cd /path/to/kneaddata_output
mkdir fastqc_out

# Loads the fastqc module
module load fastqc/0.11.7

# Peforms quality control checks on the QC'ed reads using FastQC
fastqc -t 30 *paired* -o fastqc_out/

# Loads the multiqc module
module load multiqc/1.7

# Aggregate the results of fastqc quality control checks using MultiQC
cd /path/to/fastqc_out

# Loads the multiQC program
module load multiqc/1.7

multiqc *_fastqc.zip

echo "post-QC analysis complete"

date


