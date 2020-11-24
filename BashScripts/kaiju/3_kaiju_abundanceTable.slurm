#!/bin/sh
#SBATCH --job-name=abundance_table		# Job name
#SBATCH --account=group_name			# your own sponser or account
#SBATCH --qos=group_name                # your own sponser or account
#SBATCH --mail-type=ALL               	# Mail events
#SBATCH --mail-user=name@email.com      # Where to send email
#SBATCH --ntasks=1                    	# Run on a single machine (node)
#SBATCH --cpus-per-task 30		  		# Run on a several CPU
#SBATCH --mem-per-cpu=1gb	            # Memory limit
#SBATCH --time=01:00:00               	# Time: hrs:min:sec
#SBATCH --output=output_%j.out		   	# Output and error log  

pwd; hostname; date 

# loads Kaiju
module load kaiju/1.7.2

# Merge files from all samples to a single table 

cd TaxoClassn

kaiju2table -t /path/to/kaijus/database/Directory/nodes.dmp \
		-n /path/to/kaijus/database/Directory/names.dmp \ 
		-r species -o Merged_files.tsv *.out \ 
		-c 10 -l superkingdom,phylum,class,order,family,genus,species

date
