#!/bin/sh
#SBATCH --job-name=combine_files		# Job name
#SBATCH --account=group_name			# your own sponser or account
#SBATCH --qos=group_name                # your own sponser or account
#SBATCH --mail-type=ALL               	# Mail events
#SBATCH --mail-user=name@email.com      # Where to send email	
#SBATCH --ntasks=1                    	# Run on a single machine (node)
#SBATCH --cpus-per-task 1		  		# Run on a several CPU
#SBATCH --mem-per-cpu=1gb             	# Memory limit
#SBATCH --time=02:00:00               	# Time: hrs:min:sec
#SBATCH --output=output_%j.out   		# Output and error log 

pwd; hostname; date 

# loads MetaPhlAn2 
module load metaphlan2/2.96.1

cd /path/to/metaphlan2/profiles

# Merge taxonomic profiles for each sample into a single .txt file
merge_metaphlan_tables.py *_profile.txt > merged_abundance_table.txt

date 